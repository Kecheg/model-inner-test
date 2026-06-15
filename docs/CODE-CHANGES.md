# fxp weight-share Overlay 代码变更清单

> **2026-06 更新**：本清单原基于 v0.18.0。已迁移到 **v0.22.1**，canonical 树为
> `vllm-0.22.1/`，本清单仍准确（hook 点结构在两版相同）。迁移细节见
> `docs/MIGRATION-0.22.1.md`。下面"不包含在内的"补全了之前漏记的 4 个文件。

对比基准：vLLM v0.22.1 (`docker.1ms.run/vllm/vllm-openai:v0.22.1`)
排除：kvcached 独立包

---

## 新增文件（原生 vLLM 不存在）

### 1. `vllm/config/weight_sharing.py`
`WeightSharingConfig` 数据类：
```python
mode: str          # "off" | "primary" | "secondary"
registry_path: str # 注册表目录路径
timeout_seconds: int = 300
```

### 2. `vllm/distributed/weight_sharing/__init__.py`
模块入口，导出 `WeightSharingManager`。

### 3. `vllm/distributed/weight_sharing/manager.py`
核心逻辑（~560 行）：

- `export_weights(model)` — primary 调用，对每个 shareable param 调用 `reduce_tensor` 生成 IPC handle，写入 registry JSON
- `import_weights(model)` — secondary 调用，读 registry → `ipc_tensor` → `param.data = ipc_tensor`（零拷贝）
- `_compute_model_hash(model)` — 用 model config 计算 PP-invariant 的 hash，用于 primary/secondary 校验
- `_derive_model_id()` — 从 model path 推导唯一 ID
- `_wait_for_primary()` — 轮询 `primary_ready.signal`
- `_move_cpu_buffers_to_current_device(model)` — 迁移 CPU buffer 到 CUDA（启用 CUDA graph）

### 4. `vllm/distributed/weight_sharing/parameter_filter.py`
判断参数是否可共享：

- `is_shareable_parameter(name)` — decoder layer 内的权重 → True；embed/head/norm → False
- `get_shareable_param_names(model)` / `get_non_shareable_param_names(model)`
- 排除列表：`model.embed_tokens.*`, `model.norm.*`, `lm_head.*`, `*.q_scale`, `*.k_scale` 等

### 5. `vllm/distributed/weight_sharing/registry.py`
注册表文件读写：
- 路径格式：`{base}/{model_id}/pp{pp}_tp{tp}/`
- `write_registry()` / `read_registry()`
- `mark_primary_ready()` — 创建 `primary_ready.signal`
- 文件锁 `flock(LOCK_EX)` 用于存活检测

### 6. `vllm/weight_exporter.py`
`export-weights` 后端（~210 行）：

- `run_weight_export_service()` — 加载模型 → 导出 IPC handle → 保活，不启动推理引擎
- 支持分布式 export（auto-launch 子进程，替代 torchrun）
- 不含 scheduler、KV cache、CUDA graph、HTTP server

### 7. `vllm/entrypoints/cli/export_weights.py`
`vllm export-weights` CLI 子命令（~177 行）：
- 解析 EngineArgs → 创建 VllmConfig → 调用 `run_weight_export_service`
- 支持 `--external-launcher` 分布式模式
- 自动 `--enforce-eager`

---

## 变更文件（覆盖原生）

### 8. `vllm/config/__init__.py`
```python
from vllm.config.weight_sharing import WeightSharingConfig
```

### 9. `vllm/config/vllm.py`
`VllmConfig` 增加字段：
```python
weight_sharing_config: WeightSharingConfig | None = None
```

### 10. `vllm/engine/arg_utils.py`
新增 CLI 参数：
```python
--weight-sharing-config '{"mode":"primary","registry_path":"..."}'
```

### 11. `vllm/model_executor/model_loader/base_loader.py`
`load_model()` 流程改造（~55 行变更）：

```python
# secondary 用 CPU 初始化（vs primary 用 CUDA）
init_device = "cpu" if mode == "secondary" else target_device

with init_device:
    model = initialize_model(...)     # 0 GPU 分配

self.load_weights(model, ...)         # 非共享参数加载，共享参数 no-op
process_weights_after_loading(...)     # FP8 量化后处理（shareable 层被 stub 跳过）
post_load_weights(model, ...)         # IPC 导入（替换占位符）
```

### 12. `vllm/model_executor/model_loader/default_loader.py`
新增函数（~120 行）：

| 函数 | 作用 |
|------|------|
| `_get_phase2_shareable_names(model)` | 获取可共享参数名集合 |
| `_shrink_shareable_params(model, names)` | 非共享 → CUDA；共享 → 1-element CPU 占位符 |
| `_patch_shareable_weight_loaders(model, names)` | 共享参数 weight_loader 改为 no-op |
| `_patch_shareable_quant_postprocess(model, names)` | 共享层的 `process_weights_after_loading` 替换为 stub（防 FP8 误处理），保存原始函数 |
| `post_load_weights()` | 调用 `WeightSharingManager.import_weights()` → IPC 导入 |
| `_ensure_workspace(model)` | IPC 导入后：block-FP8 层重建 `input_scale` + Marlin workspace |

### 13. `vllm/v1/worker/gpu_model_runner.py`
`load_model()` 末尾增加（~20 行）：
```python
if ws_config and ws_config.mode == "primary":
    wsm = WeightSharingManager(ws_config)
    wsm.export_weights(self.model)   # 导出所有 shareable 参数的 IPC handle
    logger.info("PRIMARY - exported IPC handles.")
```
Secondary 分支：跳过 `prepare_communication_buffer_for_model`（FP8 MoE 模块化 kernel
路径不兼容 legacy prepare）。

### 14. `vllm/v1/worker/gpu_worker.py`（之前清单遗漏）
- 模块级新增 3 个 helper：`requires_distributed_context`、`init_device_context`、
  `init_distributed_context`（被 `weight_exporter.py` / `export_multi_weights.py` 复用）。
- `init_device` 中：secondary 在分布式初始化前 `WeightSharingManager.pre_open_handles()`，
  确保 CUDA IPC handle 在正确的设备上下文下打开。
- 内存快照断言：secondary 模式放宽（IPC 权重不计入本地分配）。

### 15. `vllm/entrypoints/cli/main.py`（之前清单遗漏）
`CMD_MODULES` 注册两个 export 子命令：`export-weights`、`export-multi-weights`。

### 16. `vllm/entrypoints/cli/export_multi_weights.py`（之前清单遗漏，新增文件）
`vllm export-multi-weights` CLI 子命令：一次初始化分布式 + 循环加载多个模型并各自
`export_weights`，避免多次启停 NCCL 的开销。

### 17. `vllm/model_executor/model_loader/utils.py`（v0.22.1 无需改动）
> 旧 overlay 仅删了一段注释，无语义变化；迁移到 0.22.1 时已跳过。

---

## 不包含在内的

| 项目 | 说明 |
|------|------|
| kvcached | 独立 pip 包，vllm 运行时 monkey-patch（elastic_block_pool, engine_core, gpu_model_runner, gpu_worker, kv_cache_coordinator） |
| C/CUDA 代码 | 无。全部 Python 实现，走 PyTorch 已有的 CUDA IPC API |
