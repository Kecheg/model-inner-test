# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""跨进程模型权重共享编排器。

Primary 流程（导出端）：
  1. 完整加载模型（含 process_weights_after_loading 量化重排）
  2. 调用 export_weights(model)：为每个共享参数创建 CUDA IPC handle
     并写入文件注册表
  3. 创建 ready 信号 → secondary 可以开始导入

Secondary 流程（导入端）：
  1. CPU 初始化模型（0 GPU 峰值）
  2. 缩减共享参数为 CPU 占位符（释放 GPU 显存）
  3. 从磁盘加载非共享参数
  4. 调用 import_weights(model)：
     等待 primary 就绪 → 读取注册表 → 从 IPC handle 重建 tensor
     → param.data = ipc_tensor（零拷贝，指向 primary 的 GPU 内存）
  5. 启动 5 秒 watchdog 线程监控 primary 存活
  6. 注册 atexit 清理

关键设计：
  - CUDA IPC handle 通过 reduce_tensor/rebuild 实现零拷贝共享
  - flock 管理 primary 生命周期，覆盖 SIGKILL
  - model_hash 校验确保 secondary 连接到正确的 primary
  - TP/PP rank 隔离注册表目录
  - LoRA 运行时检测拒绝（IPC tensor 无只读保护）
"""

import hashlib
import math
import os
import threading
import time
from typing import Optional

import torch
from torch.multiprocessing.reductions import reduce_tensor  # 核心 API：创建 CUDA IPC handle

from vllm.config import get_current_vllm_config
from vllm.config.weight_sharing import WeightSharingConfig
from vllm.distributed.weight_sharing.parameter_filter import (
    filter_shareable_param_names,
    get_shareable_param_names,        # 获取可共享参数名称集合
)
from vllm.distributed.weight_sharing.registry import WeightSharingRegistry
from vllm.logger import init_logger

logger = init_logger(__name__)


class WeightSharingManager:
    """协调 Primary 导出和 Secondary 导入。"""

    # 模块级缓存：在 init_device 阶段预打开 IPC handles，
    # 存储 tensor 供后续 import_weights 使用。
    # key: "model_id:tp_rank:param_name", value: torch.Tensor
    _pre_opened: dict[str, torch.Tensor] = {}

    def __init__(self, config: WeightSharingConfig,
                 tp_rank: Optional[int] = None,
                 pp_rank: Optional[int] = None) -> None:
        self._config = config
        self._watchdog_stop: Optional[threading.Event] = None
        # 允许调用方显式传入 tp_rank/pp_rank（pre_open 阶段
        # TP/PP group 未初始化）
        if tp_rank is not None:
            self._tp_rank = tp_rank
        else:
            self._tp_rank = _get_tp_rank()
        if pp_rank is not None:
            self._pp_rank = pp_rank
        else:
            self._pp_rank = _get_pp_rank()
        # 注册表路径：{base}/{model_id}/pp{pp}_tp{tp}
        mid = _derive_model_id()
        nested = os.path.join(
            config.registry_path, mid, f"pp{self._pp_rank}_tp{self._tp_rank}")
        self._registry = WeightSharingRegistry(registry_path=nested)
        self._cleaned = False  # 幂等清理标志

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_primary(self) -> bool:
        return self._config.mode == "primary"

    def is_secondary(self) -> bool:
        return self._config.mode == "secondary"

    @torch.inference_mode()   # 禁用梯度计算，节省显存
    def export_weights(self, model: torch.nn.Module) -> None:
        """导出共享权重为 CUDA IPC handle。

        调用时机：model 加载完成且 process_weights_after_loading
        （Marlin repack 等量化后处理）已执行之后。
        """
        if not self.is_primary():
            raise RuntimeError(
                "export_weights() 只能在 'primary' 模式下调用。")

        shareable_names = get_shareable_param_names(model, self._config)
        _log_special_weight_sharing_summary(model, shareable_names, "primary")
        if not shareable_names:
            logger.warning("WeightSharing: 没有找到可共享的参数。")
            return

        # 1. 获取排他锁，确保写入注册表时没有其他 primary 竞争
        self._registry.acquire_primary_lock()

        # 2. 获取当前 GPU 的 UUID（用于 CUDA IPC handle 的 GPU 绑定校验）
        device_index = torch.cuda.current_device()
        gpu_uuid = self._get_gpu_uuid(device_index)

        # 3. 遍历每个共享参数，创建 CUDA IPC handle
        params_dict = dict(model.named_parameters())
        weights_data: dict[str, dict] = {}

        for name in sorted(shareable_names):
            param = params_dict[name]
            # 保证 tensor 在物理内存上连续，IPC handle 要求连续性
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
            # ★ 核心：reduce_tensor 创建 CUDA IPC handle
            # 返回 (rebuild_cuda_tensor, args_tuple)
            # args_tuple 包含 cudaIpcMemHandle 等序列化数据
            ipc_handle = reduce_tensor(param.data)
            # 序列化为 base64 字符串，便于 JSON 存储
            serialized = WeightSharingRegistry.serialize_ipc_handle(ipc_handle)

            weights_data[name] = {
                "dtype": str(param.data.dtype).split(".")[-1],
                "shape": list(param.data.shape),
                gpu_uuid: serialized,          # UUID 作为 key，支持多 GPU 拓扑
            }

        # 4. 写入注册表（含 model_hash / tp_rank / tp_size 等元数据）
        registry_data = {
            "model_hash": _compute_model_hash(model),  # secondary 校验用
            "primary_pid": os.getpid(),
            "primary_device_index": device_index,
            "tp_rank": _get_tp_rank(),                  # secondary 校验 TP 拓扑
            "tp_size": _get_tp_size(),
            "gpu_uuids": {gpu_uuid: device_index},
            "timestamp": time.time(),
            "weights": weights_data,
        }

        self._registry.write_registry(registry_data)
        self._registry.signal_ready()  # 创建 ready 信号文件
        shared_bytes = sum(
            int(math.prod(weight_info["shape"]))
            * torch.empty((), dtype=_dtype_from_string(weight_info["dtype"]))
            .element_size()
            for weight_info in weights_data.values())
        logger.info(
            "WeightSharing primary: 导出 %d 个共享参数的 IPC handle "
            "(GPU UUID: %s, size=%.3f GiB)。", len(weights_data), gpu_uuid,
            shared_bytes / 1024**3)

    def pre_open_handles(self, device_index: int) -> int:
        """预打开所有 IPC handles（必须在任何 CUDA 操作之前调用）。

        cudaIpcOpenMemHandle 仅在进程 CUDA context 完全干净时才能成功。
        device_index 由调用方提供，避免调用任何 CUDA API（会污染 context）。
        打开的 tensor 缓存在类级 dict 中，供 import_weights 使用。
        返回预打开的 handle 数量。
        """
        if not self.is_secondary():
            return 0

        # 等待 primary 就绪并读取注册表
        if not self._registry.wait_for_primary(self._config.timeout_seconds):
            raise RuntimeError(
                "WeightSharing pre_open: 等待 primary 超时。"
                "请确认 primary 已先启动且注册表目录可访问。")

        registry_data = self._registry.read_registry()
        registry_weights: dict[str, dict] = registry_data.get("weights", {})
        weights_data = {
            name: registry_weights[name]
            for name in filter_shareable_param_names(
                registry_weights.keys(), self._config)
        }

        # 从注册表数据中找到对应 device_index 的 GPU UUID，
        # 直接从 primary 导出的数据中取第一个匹配的 UUID
        # (避免调用 CUDA API)
        gpu_uuid = self._find_gpu_uuid_from_registry(
            device_index, registry_data)
        if gpu_uuid is None:
            raise RuntimeError(
                f"WeightSharing pre_open: 找不到 device_index={device_index} "
                f"对应的 GPU UUID in registry。")

        cache_prefix = f"{_derive_model_id()}:{self._tp_rank}"
        opened = 0

        for name, weight_info in weights_data.items():
            serialized = self._find_ipc_handle(
                name, weight_info, gpu_uuid, registry_data)
            if serialized is None:
                raise RuntimeError(
                    f"WeightSharing pre_open: GPU UUID 不匹配 {name}。")

            tensor = self._reconstruct_tensor(serialized, device_index, name)
            if tensor is None:
                raise RuntimeError(
                    f"WeightSharing pre_open: 无法重建 IPC tensor {name}。"
                    f"primary 可能已退出或 GPU 内存不可访问。")

            WeightSharingManager._pre_opened[
                f"{cache_prefix}:{name}"] = tensor
            opened += 1

        logger.info(
            "WeightSharing: pre-opened %d/%d IPC handles (device=%d).",
            opened, len(weights_data), device_index)
        return opened

    @torch.inference_mode()
    def import_weights(self, model: torch.nn.Module) -> set[str]:
        """从注册表导入共享权重，替换本地参数数据。

        Secondary 调用。每个共享参数通过 IPC handle 重建 tensor，
        直接指向 primary 的 GPU 内存（零拷贝）。

        返回成功导入的参数名称集合。
        """
        if not self.is_secondary():
            raise RuntimeError(
                "import_weights() 只能在 'secondary' 模式下调用。")

        # ====== 安全检查 ======

        # 拒绝 LoRA：LoRA 融合是 in-place 操作，会通过 IPC tensor 写入 primary
        # 的 GPU 内存，造成权重污染。
        if _has_lora(model):
            raise RuntimeError(
                "WeightSharing: 权重共享不兼容 LoRA。"
                "LoRA 会原地修改 base weight，可能破坏 primary 的共享 GPU 内存。")

        # 等待 primary 就绪
        if not self._registry.wait_for_primary(self._config.timeout_seconds):
            raise RuntimeError(
                "WeightSharing: 等待 primary 进程超时。"
                "请确认 primary 已先启动且注册表目录可访问。")

        # 读取注册表
        registry_data = self._registry.read_registry()

        # ====== 模型兼容性验证 ======

        # 1. model_hash：确保是同一模型
        local_hash = _compute_model_hash(model)
        remote_hash = registry_data.get("model_hash", "")
        if local_hash != remote_hash:
            raise RuntimeError(
                f"WeightSharing: 模型 hash 不匹配。"
                f"本地 hash={local_hash}，primary hash={remote_hash}。"
                f"请确认本地和 primary 使用相同的模型。")

        # 2. TP rank：确保 secondary 从正确的 primary rank 导入
        local_rank = _get_tp_rank()
        remote_rank = registry_data.get("tp_rank", -1)
        if local_rank != remote_rank:
            raise RuntimeError(
                f"WeightSharing: TP rank 不匹配。"
                f"本地 rank={local_rank}，primary rank={remote_rank}。")

        # 3. TP size：不同 TP 大小参数形状不兼容
        remote_tp = registry_data.get("tp_size", -1)
        local_tp = _get_tp_size()
        if local_tp != remote_tp:
            raise RuntimeError(
                f"WeightSharing: TP 大小不匹配。"
                f"本地 TP={local_tp}，primary TP={remote_tp}。")

        # ====== 开始导入 ======

        registry_weights: dict[str, dict] = registry_data.get("weights", {})
        expected_names = get_shareable_param_names(model, self._config)
        missing_expected = expected_names - set(registry_weights)
        if missing_expected:
            raise RuntimeError(
                "WeightSharing: primary 未导出 secondary 配置要求共享的参数。"
                f"缺失 {len(missing_expected)} 个，examples="
                f"{sorted(missing_expected)[:10]}。请确保 primary 共享集合是 "
                "secondary 共享集合的超集。")
        weights_data = {
            name: registry_weights[name]
            for name in expected_names
        }
        _log_special_weight_sharing_summary(
            model, set(weights_data.keys()), "secondary")
        device_index = torch.cuda.current_device()
        gpu_uuid = self._get_gpu_uuid(device_index)

        params_dict = dict(model.named_parameters())
        imported: set[str] = set()
        cache_prefix = f"{_derive_model_id()}:{self._tp_rank}"

        for name, weight_info in weights_data.items():
            if name not in params_dict:
                logger.warning(
                    "WeightSharing: 参数 %s 在本地模型中未找到，跳过。", name)
                continue

            # 优先从预打开缓存获取（init_device 阶段已打开）
            cache_key = f"{cache_prefix}:{name}"
            if cache_key in WeightSharingManager._pre_opened:
                ipc_tensor = WeightSharingManager._pre_opened.pop(cache_key)
            else:
                # 回退：从序列化 handle 重建（旧路径，需要 clean CUDA context）
                serialized = self._find_ipc_handle(
                    name, weight_info, gpu_uuid, registry_data)
                if serialized is None:
                    raise RuntimeError(
                        f"WeightSharing: GPU UUID 不匹配 {name}。"
                        f"primary 和 secondary 必须在同一物理 GPU 上。")
                ipc_tensor = self._reconstruct_tensor(
                    serialized, device_index, name)
                if ipc_tensor is None:
                    raise RuntimeError(
                        f"WeightSharing: 无法重建 IPC tensor {name}。"
                        f"primary 可能已退出或 GPU 内存不可访问。")

            # 替换 param.data → 直接指向 primary 的 GPU 内存
            param = params_dict[name]
            param.data = ipc_tensor
            imported.add(name)

        self._move_cpu_buffers_to_current_device(model)

        # ====== 导入后验证 ======
        if len(imported) != len(weights_data):
            missing = len(weights_data) - len(imported)
            raise RuntimeError(
                f"WeightSharing: 只导入 {len(imported)}/{len(weights_data)} "
                f"secondary 配置要求的共享参数（{missing} 个缺失），无法运行。")

        # 释放 PyTorch CUDA 缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        imported_bytes = sum(
            int(math.prod(weights_data[name]["shape"]))
            * torch.empty((), dtype=_dtype_from_string(
                weights_data[name]["dtype"])).element_size()
            for name in imported)
        logger.info(
            "WeightSharing secondary: 导入 %d/%d 个共享参数 (size=%.3f GiB)。",
            len(imported), len(weights_data), imported_bytes / 1024**3)

        # ====== 启动后台监控 ======
        # CUDA IPC tensor 只在 primary 持有 GPU 内存时有效。
        # 启动 daemon 线程每 5s 检查一次 primary 存活状态。
        # 如果 primary 死亡（flock 释放），立即 os._exit(127) 退出，
        # 防止在悬空 GPU 指针上继续推理（导致静默错误或 CUDA 崩溃）。
        self._start_watchdog()

        # 注册 atexit——正常退出时释放锁和注册表文件
        import atexit
        atexit.register(self.cleanup)
        return imported

    def cleanup(self) -> None:
        """停止 watchdog 并释放资源（幂等：可多次调用）。"""
        if self._cleaned:
            return
        self._cleaned = True
        self._stop_watchdog()
        if self.is_primary():
            self._registry.cleanup()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gpu_uuid(device_index: int) -> str:
        """获取指定 GPU 设备的 UUID（用于 IPC handle 绑定校验）。"""
        return str(torch.cuda.get_device_properties(device_index).uuid)

    def _start_watchdog(self) -> None:
        """启动后台监控线程，定期检查 primary 是否存活。

        检测间隔 5 秒——足够短以减少悬空指针的风险窗口，
        足够轻量以忽略开销（一次 flock syscall）。
        """
        if not self.is_secondary():
            return
        if self._watchdog_stop is not None:
            return  # 已在运行
        self._watchdog_stop = threading.Event()
        interval = 5  # 每 5 秒检查一次

        def _watch():
            while not self._watchdog_stop.wait(interval):
                if not self._registry.is_primary_alive():
                    logger.critical(
                        "WeightSharing WATCHDOG: primary 进程已死亡！"
                        "CUDA IPC tensor 已失效，立即退出以防数据损坏。")
                    os._exit(127)  # 硬退出——此时任何优雅步骤都可能访问悬空内存

        thread = threading.Thread(target=_watch, daemon=True, name="ws-watchdog")
        thread.start()
        logger.debug("WeightSharing watchdog 已启动（interval=%ds）。", interval)

    def _stop_watchdog(self) -> None:
        """停止 watchdog 线程。"""
        if self._watchdog_stop is not None:
            self._watchdog_stop.set()
            self._watchdog_stop = None

    def _find_gpu_uuid_from_registry(
        self, device_index: int, registry_data: dict,
    ) -> Optional[str]:
        """从注册表数据中获取与 device_index 匹配的 GPU UUID。

        在没有 CUDA context 时使用——从 primary 导出的数据中，
        根据权重信息推断当前 device_index 对应的 GPU UUID。
        跳过非 UUID 字段（dtype、shape），只返回真正的 GPU UUID。
        """
        weights_data: dict = registry_data.get("weights", {})
        for name, weight_info in weights_data.items():
            for key in weight_info:
                if key in ("dtype", "shape"):
                    continue
                return str(key)
        return None

    def _find_ipc_handle(
        self, name: str, weight_info: dict, gpu_uuid: str, registry_data: dict,
    ) -> Optional[str]:
        """根据 GPU UUID 匹配对应物理 GPU 的 IPC handle。

        CUDA IPC handle 是 GPU 绑定的——不能跨不同物理 GPU 使用。
        """
        if gpu_uuid in weight_info:
            return weight_info[gpu_uuid]  # type: ignore[return-value]

        logger.error(
            "WeightSharing: GPU UUID 不匹配 %s "
            "(当前 %s，primary 导出: %s)。"
            "primary 和 secondary 必须在同一物理 GPU 上。",
            name, gpu_uuid,
            [k for k in weight_info if k.startswith("GPU-")])
        return None

    @staticmethod
    def _reconstruct_tensor(
        serialized: str, device_index: int, name: str,
    ) -> Optional[torch.Tensor]:
        """从序列化的 IPC handle 重建 CUDA tensor。

        IPC handle args 中的 device_index（位置 6）需要 patch 为
        当前进程的实际设备索引，参考 IPCWeightTransferEngine。
        """
        try:
            # 反序列化 IPC handle
            ipc_handle = WeightSharingRegistry.deserialize_ipc_handle(
                serialized)
            func, args = ipc_handle
            list_args = list(args)
            # index 6 is the CUDA storage device used by PyTorch's
            # CUDA IPC rebuild path. PyTorch 2.10 still expects an int here.
            list_args[6] = device_index
            torch.cuda.set_device(device_index)
            torch.cuda._lazy_init()
            logger.info(
                "WeightSharing: rebuild IPC tensor %s on cuda:%d "
                "(current=%d, visible_devices=%d, storage_device_arg=%r)",
                name, device_index, torch.cuda.current_device(),
                torch.cuda.device_count(), list_args[6])
            # ★ 核心：重建 tensor，直接指向 primary 的 GPU 物理内存。
            # 重建 IPC tensor 时必须确保当前 CUDA device 与 handle 的
            # device_index 一致，否则 secondary 在单卡/TP 场景都可能把
            # handle 打开到错误上下文。
            return func(*list_args)
        except Exception as exc:
            logger.error(
                "WeightSharing: 重建 tensor 失败 %s: %s", name, exc)
            return None

    @staticmethod
    def _move_cpu_buffers_to_current_device(model: torch.nn.Module) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        moved = 0
        for name, buf in model.named_buffers():
            if not isinstance(buf, torch.Tensor) or buf.device.type != "cpu":
                continue
            module_name, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            module._buffers[buffer_name] = buf.to(device=device)
            moved += 1

        if moved:
            logger.info(
                "WeightSharing secondary: moved %d CPU buffers to %s.",
                moved, device)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _log_special_weight_sharing_summary(
    model: torch.nn.Module,
    shared_names: set[str],
    role: str,
) -> None:
    params = dict(model.named_parameters())
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "embed_tokens": [],
        "lm_head": [],
        "norm": [],
        "vision_encoder": [],
        "projector_connector": [],
    }
    for name, param in params.items():
        # Embed tokens: decoder-only (embed_tokens, wte) and encoder (word_embeddings)
        if "embed_tokens" in name or name.endswith(".wte.weight") \
                or "word_embeddings" in name:
            groups["embed_tokens"].append((name, param))
        # LM head / pooler / classifier head
        elif "lm_head" in name or "embed_out" in name \
                or "pooler" in name or "score" in name \
                or "classifier" in name:
            groups["lm_head"].append((name, param))
        # Norm: decoder RMSNorm/LayerNorm and encoder LayerNorm
        elif name == "model.norm.weight" or name == "model.norm.bias" or \
                name.endswith(".ln_f.weight") or name.endswith(".ln_f.bias") or \
                name.endswith(".final_layer_norm.weight") or \
                name.endswith(".final_layer_norm.bias") or \
                "embeddings.LayerNorm" in name:
            groups["norm"].append((name, param))
        # Vision encoder / image encoder / visual backbone
        elif "vision" in name or "visual" in name or "image_encoder" in name \
                or "vision_model" in name or "vision_tower" in name:
            groups["vision_encoder"].append((name, param))
        # Projector / connector between vision encoder and language model
        elif "projector" in name or "connector" in name \
                or "mm_projector" in name or "multi_modal_projector" in name:
            groups["projector_connector"].append((name, param))

    for group_name, items in groups.items():
        if not items:
            logger.info(
                "WeightSharing %s: %s not found; shared=no size=0.000 GiB.",
                role, group_name)
            continue
        total_bytes = sum(_tensor_nbytes(param.data) for _, param in items)
        shared = any(name in shared_names for name, _ in items)
        logger.info(
            "WeightSharing %s: %s shared=%s params=%d size=%.3f GiB names=%s",
            role, group_name, shared, len(items), total_bytes / 1024**3,
            [name for name, _ in items])


def _dtype_from_string(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype == "float8_e4m3fn":
        return torch.float8_e4m3fn
    if dtype == "float8_e5m2":
        return torch.float8_e5m2
    if dtype == "int8":
        return torch.int8
    if dtype == "uint8":
        return torch.uint8
    raise ValueError(f"Unsupported dtype string: {dtype}")


# ============================================================================
# 模块级工具函数
# ============================================================================

def _compute_model_hash(model: torch.nn.Module) -> str:
    """计算模型架构 hash，用于 secondary 校验 primary 是否兼容。

    使用模型路径 + 总参数数量 + dtype 计算 hash。
    模型路径在所有 PP/TP rank 间一致，参数数量和 dtype 提供额外校验。
    不使用具体参数名（不同 PP stage 的参数名不同）。
    """
    h = hashlib.sha256()
    try:
        cfg = get_current_vllm_config()
        model_path = cfg.model_config.model
    except Exception:
        model_path = "unknown"
    h.update(model_path.encode())
    params = dict(model.named_parameters())
    h.update(str(len(params)).encode())            # 总参数数
    if params:
        # 取第一个参数的 dtype 作为模型整体 dtype 的近似校验
        first_p = next(iter(params.values()))
        h.update(str(first_p.data.dtype).encode())
    return h.hexdigest()[:16]


def _derive_model_id() -> str:
    """从模型路径自动推导唯一标识。

    格式：{模型目录名}-{SHA256 前 12 位}
    例：Qwen3-4B-Thinking-2507-FP8-safe-29d474a14a55
    """
    try:
        cfg = get_current_vllm_config()
        model = cfg.model_config.model
    except Exception:
        logger.exception("WeightSharing: 无法读取 model config，回退到 'default'")
        return "default"
    h = hashlib.sha256(model.encode()).hexdigest()[:12]
    safe = model.rstrip("/").rsplit("/", 1)[-1].replace(" ", "_")
    return f"{safe}-{h}"


def _get_tp_rank() -> int:
    """获取 Tensor Parallelism rank。"""
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )
        return get_tensor_model_parallel_rank()
    except Exception:
        logger.exception("WeightSharing: 无法读取 TP rank，回退到 0")
        return 0


def _get_pp_rank() -> int:
    """获取 Pipeline Parallelism rank。"""
    try:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().rank_in_group
    except Exception:
        return 0


def _get_tp_size() -> int:
    """获取 Tensor Parallelism world size。"""
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _has_lora(model: torch.nn.Module) -> bool:
    """检测模型是否使用了 LoRA 适配器。

    LoRA 与权重共享不兼容，因为 LoRA 融合可能原地修改权重，
    通过 IPC tensor 写入 primary 的 GPU 内存会造成污染。
    """
    for module in model.modules():
        module_type = type(module).__name__
        if "LoRA" in module_type or "lora" in module_type.lower():
            return True
    return False
