# vLLM Weight Sharing — fxp Overlay

在 vLLM v0.18.0 上实现跨进程 GPU 权重共享。两个 vLLM 实例共享 decoder 层权重，
secondary 只加载非共享参数（embed/head/norm），节省显存。

## 目录结构

```
model-inner-test/
├── README.md                           # 本文
├── Dockerfile                          # 构建 weight-share 镜像
├── docs/
│   ├── DEPLOY-GUIDE.md                 # 部署指南（单卡 + PP=4）
│   ├── CODE-CHANGES.md                 # 代码变更清单（vs 原生 vLLM）
│   └── BENCHMARK-RESULTS.md            # 性能测试结果
├── scripts/
│   ├── start-primary-1gpu.sh           # 单卡 Primary 启动
│   ├── start-secondary-1gpu.sh         # 单卡 Secondary 启动（参数：端口 显存利用率）
│   ├── start-primary-pp4.sh            # PP=4 Primary 启动
│   └── start-secondary-pp4.sh          # PP=4 Secondary 启动
└── vllm/                               # fxp 修改的源码（覆盖到 site-packages）
```

## 快速开始

### 1. 构建镜像

```bash
cd model-inner-test
docker build -t vllm-weight-share:v0.18.0 -f Dockerfile .
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

详细部署步骤见 `docs/DEPLOY-GUIDE.md`。
