# vLLM Weight Sharing — fxp Overlay

在 vLLM **v0.22.1**（基础镜像 `docker.1ms.run/vllm/vllm-openai:v0.22.1`）上
实现跨进程 GPU 权重共享。两个 vLLM 实例共享 decoder 层权重，secondary 只加载
非共享参数（embed/head/norm），节省显存。

> **历史**：原始 overlay 基于 vLLM 0.18.0（仍保留在 `vllm/` 目录作 diff 参照）；
> 2026-06 已迁移到 0.22.1，canonical 树为 `vllm-0.22.1/`。详见
> `docs/MIGRATION-0.22.1.md`（含完整端到端冒烟测试报告）。

## 目录结构

```
model-inner-test/
├── README.md                           # 本文
├── Dockerfile                          # 构建 weight-share 镜像
├── docs/
│   ├── DEPLOY-GUIDE.md                 # 部署指南（单卡 + PP=4）
│   ├── CODE-CHANGES.md                 # 代码变更清单（vs 原生 vLLM）
│   ├── MIGRATION-0.22.1.md             # 0.18→0.22.1 迁移记录 + 验证报告
│   └── BENCHMARK-RESULTS.md            # 性能测试结果
├── scripts/
│   ├── start-primary-1gpu.sh           # 单卡 Primary 启动
│   ├── start-secondary-1gpu.sh         # 单卡 Secondary 启动（参数：端口 显存利用率）
│   ├── start-primary-pp4.sh            # PP=4 Primary 启动
│   └── start-secondary-pp4.sh          # PP=4 Secondary 启动
├── tests/                              # 端到端冒烟测试（00 env→ 05 sardeenz e2e）
├── vllm-0.22.1/                        # canonical 树（0.22.1 + weight-sharing overlay）
├── vllm/                               # 旧 overlay（0.18，保留作 diff 参照）
├── kvcached/                           # kvcached 完整源码（含 0.22.1 适配修复）
└── sardeenz/                           # Sardeenz 完整源码，用于服务启动和调度
```

## 快速开始

### 1. 构建镜像

```bash
cd model-inner-test
docker build -t vllm-weight-share:v0.22.1 -f Dockerfile .
```

### 2. 部署（以单卡为例）

```bash
# 终端 1: 启动 Primary
bash scripts/start-primary-1gpu.sh

# 终端 2: 启动 Secondary
bash scripts/start-secondary-1gpu.sh 8000 0.60
```

### 3. 验证

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-FP8","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 关键约束

- Primary 和 Secondary 必须在**同一张物理 GPU**上运行（CUDA IPC 要求）
- 容器启动需要 `--security-opt seccomp=unconfined`
- JSON 参数用单引号包裹，变量用 `'"${VAR}"'` 拼接，不要用 `\"` 转义
- TP2 + 权重复用场景需要容器使用 host PID/IPC namespace，否则 secondary 无法稳定打开 primary 导出的 CUDA IPC handle。
- kvcached 的 CUDA extension 需要在最终镜像的 torch/CUDA toolchain 下编译；当前 Dockerfile 已复制完整源码到 `/opt/kvcached`，如基础镜像包含匹配编译环境，可启用 Dockerfile 中的 `pip install -e /opt/kvcached`。

## 本次集成内容

- vLLM weight-sharing overlay 保持原有结构。
- 修复 TP/多卡 secondary 重建 IPC tensor 前未切换 CUDA device 导致的 `cudaErrorInvalidDeviceContext`。
- 纳入完整 kvcached 源码，保留 Sardeenz 调度侧调用 kvcached autopatch 所需的代码。
- 纳入完整 Sardeenz 源码，便于后续镜像统一构建服务启动和调度链路。
- 不包含旧的 vLLM 原生 KV tensor 动态扩缩实验改动，也不包含 `cumem_allocator.cpp` 的 sleep/wake 迁移实验改动。

详细部署步骤见 `docs/DEPLOY-GUIDE.md`。
