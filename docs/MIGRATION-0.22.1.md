# vLLM 0.18.0 → 0.22.1 迁移执行清单

> 完整背景与策略见已批准的工作计划。本文件用于跟踪落地进度。
> canonical 树：`vllm-0.22.1/`（weight-sharing 改动直接套用进此树）；旧 `vllm/`（0.18 overlay）保留作 diff 参照。
> 提取语义补丁：`diff /root/zly/vllm-0.18.0/vllm/<f>  vllm/<f>`。

## 真实 overlay 清单（以 Dockerfile 为准，docs/CODE-CHANGES.md 已过时）

新增 7：`config/weight_sharing.py`、`distributed/weight_sharing/{__init__,manager,parameter_filter,registry}.py`、`entrypoints/cli/export_weights.py`、`entrypoints/cli/export_multi_weights.py`、`weight_exporter.py`
改动 9：`config/__init__.py`、`config/vllm.py`、`engine/arg_utils.py`、`model_executor/model_loader/base_loader.py`、`model_executor/model_loader/default_loader.py`、`model_executor/model_loader/utils.py`、`v1/worker/gpu_model_runner.py`、`v1/worker/gpu_worker.py`、`entrypoints/cli/main.py`

## 进度

### Phase 0 — 测试脚手架
- [x] `tests/`：README、run_all.sh、lib/common.sh、00~05 冒烟脚本（bash -n 通过）

### Phase 1 — weight-sharing 套用到 0.22.1
- [x] 1A 拷贝 7 个新增文件（import 符号已逐个核对存在于 0.22.1）
- [x] 1B 低风险 5：config/__init__、config/vllm、arg_utils、cli/main（loader/utils 仅注释，跳过）
- [x] 1B 中风险 4：base_loader、default_loader、gpu_model_runner、gpu_worker（gpu_worker 三 helper 已连带移植；marlin/fp8-moe helper 路径已确认存在）
- [x] 1C 验证 `tests/00`→`tests/03` **全部 PASS**（镜像内 + GPU 实跑）

### Phase 2 — kvcached 适配（实测风险远低于初判）
> **重要修正**：用 0.18（kvcached 现可用）对比 0.22.1（而非 Agent 误用的远古版本）后，
> 大量"断点"为误报：`BlockPool.__init__` 两版**完全一致**；`KVCacheEvent` 两版都已在
> `vllm.distributed.kv_events`；`set_kv_cache_layout`/coordinator 等仅非破坏性变化。
> `VersionRange` 用 `packaging.version` 语义比较，0.22.1 正确命中 V9_PLUS/V10/ALL。

- [x] **唯一真实断点**：`GPUModelRunner._reshape_kv_cache_tensors` 0.22.1 移除首参 `kv_cache_config`
      → 用 `inspect.signature` 检测签名，新版从 `self.kv_cache_config` 取配置（`patches.py` 已改，版本无关）
- [x] `_allocate_kv_cache_tensors` 签名未变（无需改）；coordinator `__init__` 新增 `max_num_batched_tokens`
      经 `*args/**kwargs` 透传，非破坏
- [x] 验证 `tests/04` **PASS**（镜像内 + GPU；serve 启动无 patch 异常、chat 推理正常）

### Phase 3 — sardeenz + Dockerfile
- [x] model-manager.ts 依赖的 CLI 参数（kv_cache_memory_bytes/enable_prefix_caching/
      enable_sleep_mode/enforce_eager/...）在 0.22.1 **均为现存 config 字段**（auto-gen），**无需改 sardeenz**
- [x] vllm-entrypoint.py 依赖的 `entrypoints.cli.main.main`、`openai.api_server.build_app` 在 0.22.1 存在 → 无需改
- [x] Dockerfile：基础镜像 → `docker.1ms.run/vllm/vllm-openai:v0.22.1`；COPY 源 → `vllm-0.22.1/vllm/`；
      移除 utils.py；kvcached 路径动态推导；**py_compile 自检扩展到全部 16 overlay 文件（build 期语法门禁）**
- [x] `tests/00-04` 全 PASS；`tests/05` SKIP（需 sardeenz dist 构建产物，本次未触发）

### Phase 4 — 收尾
- [x] 更新 README / CODE-CHANGES（补全遗漏 4 文件 + 标注 0.22.1）
- [ ] 退役旧 `vllm/`（推荐保留一段时间作 diff 参照；待你确认后删除）

### 验证报告
| 测试 | 范围 | 结果 |
|---|---|---|
| `tests/00_env_check` | overlay 完整性、CLI 注册、autopatch import | ✅ PASS |
| `tests/01_weight_share_1gpu` | 单卡 export+secondary+chat（GPU 7, util=0.50） | ✅ PASS |
| `tests/02_weight_share_pp4` | PP=4 分布式 export+IPC+chat（GPU 0-3） | ✅ PASS（首次即过） |
| `tests/03_dual_secondary` | 同卡双 secondary 共享同一 primary（util=0.50/0.42） | ✅ PASS |
| `tests/04_kvcached` | kvcached monkey-patch attach + chat | ✅ PASS（trim 路由 SKIP，需 sardeenz entrypoint） |
| `tests/05_sardeenz_e2e` | sardeenz 编排链路 | ⏸️ SKIP（缺 sardeenz dist） |
| `tests/06_kvcached_with_ws` | **kvcached + weight-sharing 共存（生产真实组合）** | ✅ PASS |
| `tests/07_weight_share_tp2` | **TP=2 weight-sharing**（每 rank 独立 registry + IPC） | ✅ PASS |

### 关键发现
- weight-sharing 9 改动文件 + 7 新增文件全部正确套用到 0.22.1。
- kvcached 真实断点只有一处：`_reshape_kv_cache_tensors` 在 0.22.1 移除首参 `kv_cache_config`，
  已用 `inspect.signature` 版本无关方式修复（`kvcached/kvcached/integration/vllm/patches.py`）。
- sardeenz 依赖的 CLI 参数 / entrypoint 路径在 0.22.1 仍存在，**sardeenz 代码无需改动**。
- **0.22.1 secondary 内存约束比 0.18 严格**：`available_kv_cache = util*total − non_kv_cache_memory`，
  且 non_kv_cache 包含了 IPC 共享权重的"虚拟占用"（profile 看到）；
  secondary 部署 util 需按此重新估算（详见各 test 脚本顶部注释）。

## 关键前置/风险
- 0.22.1 docker 基础镜像 tag 待确认（备选：源码构建）
- FP8/Marlin/MoE helper 在 0.22 路径变动 → 1A/1B 逐个 import 验证
- `weight_exporter.py` 依赖 gpu_worker 的 `init_device_context/init_distributed_context/requires_distributed_context`
