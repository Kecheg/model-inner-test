# tests/ — 端到端冒烟测试

本目录把项目三层能力（weight-sharing / kvcached / sardeenz）的验证流程固化为可重复脚本，
用于保证 **vLLM 0.18.0 → 0.22.1 迁移后功能不退化**。

## 运行前提

> ⚠️ 宿主机不含 torch/vllm。**所有脚本必须在构建好的镜像内、且有 GPU 的容器中运行。**

```bash
# 1) 构建镜像（迁移完成后基础镜像应为 0.22.1）
docker build -t vllm-weight-share:v0.22.1 -f Dockerfile .

# 2) 起容器（CUDA IPC 要求 host ipc/pid；weight-sharing 要求同物理 GPU）
docker run -it --rm --runtime nvidia --ipc=host --pid=host --network host \
  --security-opt seccomp=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -v /model:/model -v "$PWD":/workspace -w /workspace \
  --entrypoint bash vllm-weight-share:v0.22.1

# 3) 容器内执行
bash tests/run_all.sh
```

## 环境变量（均有默认值，见 `lib/common.sh`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL` | `/model/models/Qwen3-8B-FP8` | 测试模型路径 |
| `SERVED_NAME` | `basename(MODEL)` | served-model-name |
| `REG_ROOT` | `/tmp/ws-registry` | 权重共享 registry 根目录 |
| `MAX_MODEL_LEN` | `2048` | 上下文长度 |
| `LOG_DIR` | `/tmp/ws-smoke-logs` | 各用例日志输出 |
| `READY_TIMEOUT` | `300` | 等待 serve/export 就绪秒数 |
| `GPU` / `GPUS` | `0` / `0,1,2,3` | 使用的 GPU |
| `SARDEENZ_TOKEN` | 空 | 05 用例 admin Bearer token（如启用鉴权） |
| `ONLY` | 空 | 只跑指定用例，如 `ONLY="00 01"` |
| `CONTINUE_ON_FAIL` | `0` | 失败后是否继续 |

## 用例清单

| 脚本 | 覆盖 | 判定标准 |
|------|------|----------|
| `00_env_check.sh` | 环境/overlay 完整性（秒级，不加载模型） | vllm 版本 0.22.x、`WeightSharingConfig` 可导入、`is_shareable_parameter` 正确、CLI 含 `export-weights`/`export-multi-weights`、kvcached autopatch 可 import、GPU 可见 |
| `01_weight_share_1gpu.sh` | 单卡权重共享 | export 出 registry+signal；secondary 日志 `IPC-imported` + startup；`/v1/models` 可用；chat 非空 |
| `02_weight_share_pp4.sh` | PP=4 权重共享 | 4 个 `primary_ready.signal`；secondary `IPC-imported`；chat 非空（GPU<4 自动 SKIP） |
| `03_dual_secondary.sh` | 同 GPU 双 secondary 共享同一 primary | 两端 `IPC-imported` + chat 非空 |
| `04_kvcached.sh` | kvcached monkey-patch attach | serve 无 patch 异常栈；chat 非空；`POST /kvcached/trim` 返回 ok |
| `05_sardeenz_e2e.sh` | sardeenz 编排链路 | backend `/health`；`/api/models/load` 触发；模型进入 running；经代理 chat 非空 |
| `06_kvcached_with_ws.sh` | **kvcached + weight-sharing 共存（生产真实组合）** | secondary 同时启用两者；日志无 kvcached patch 异常；`IPC-imported` 出现；chat 非空 |
| `07_weight_share_tp2.sh` | **TP=2 weight-sharing** | export 生成 `pp0_tp0`+`pp0_tp1` 两套 registry；secondary `IPC-imported` 出现 ≥TP 次；chat 非空 |

退出码约定：`0=PASS`、`77=SKIP`（前提缺失，如模型/GPU/构建产物不存在）、其它=`FAIL`。

## 与现有组件级单测的关系

本目录聚焦**端到端冒烟**。组件级单测保持原位，作为补充：

- `kvcached/tests/*.py`（pytest）：`cd kvcached && pytest tests/`
- `sardeenz/apps/backend/tests/*.ts`（vitest）：`cd sardeenz/apps/backend && npm test`

迁移过程中，端到端冒烟用于逐层 gating（见 `docs/MIGRATION-0.22.1.md`）。
