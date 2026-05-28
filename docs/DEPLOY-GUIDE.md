# vLLM Weight Sharing 部署指南

基于 vLLM v0.18.0 + weight-share overlay，实现两个 vLLM 实例共享同一份 decoder 层权重，节省 GPU 显存。

## 0. 环境准备

### 镜像

```bash
IMAGE=docker.1ms.run/vllm/vllm-openai:v0.18.0  # 原版镜像，需 overlay 代码
IMAGE=vllm-weight-share:qwen3-8b-fp8-pp4        # 自建镜像（含 fxp overlay）
```

### 宿主机目录约定

```bash
FXP_HOST=/sf/data/local/fxp
CODE_HOST=/sf/data/local/fxp/vllm-weight-share
```

### 容器启动模板

```bash
docker run -d --name <name> \
  --runtime nvidia --ipc=host --network host \
  --security-opt seccomp=unconfined \
  --entrypoint sleep \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v ${FXP_HOST}:/workspace/fxp \
  -v /model:/model \
  -w /workspace ${IMAGE} 10d
```

### 覆盖 fxp 代码（如果用原版镜像）

```bash
tar --exclude='*.so' --exclude='*.pyc' --exclude='__pycache__' \
  -C ${CODE_HOST} -cf - vllm \
  | docker exec -i <container> bash -c \
      'cd /usr/local/lib/python3.12/dist-packages && tar -xf -'
```

---

## 1. 单卡部署

### 1.1 启动 Primary / Exporter

```bash
REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_1gpu
MODEL=/model/models/Qwen3-8B-FP8

rm -rf ${REG_DIR}/*
mkdir -p ${REG_DIR}

CUDA_VISIBLE_DEVICES=0 vllm export-weights ${MODEL} \
  --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
  --enforce-eager \
  --max-model-len 2048 \
  > /tmp/export.log 2>&1 &

# 校验
grep "export complete" /tmp/export.log
find ${REG_DIR} -type f | sort
# 预期输出：pp0_tp0/primary_ready.signal + weight_registry.json
```

### 1.2 启动 Secondary / Importer

```bash
REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_1gpu
MODEL=/model/models/Qwen3-8B-FP8

CUDA_VISIBLE_DEVICES=0 vllm serve ${MODEL} \
  --host 0.0.0.0 --port 8000 \
  --served-model-name Qwen3-8B-FP8 \
  --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
  --gpu-memory-utilization 0.60 \
  --max-model-len 2048 \
  > /tmp/serve.log 2>&1 &

# 校验
grep "IPC-imported" /tmp/serve.log
grep "Application startup complete" /tmp/serve.log
curl -s http://127.0.0.1:8000/v1/models
```

### 1.3 再起一个 Secondary（同 GPU 共享权重）

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ${MODEL} \
  --host 0.0.0.0 --port 8001 \
  --served-model-name Qwen3-8B-FP8 \
  --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
  --gpu-memory-utilization 0.30 \
  --max-model-len 2048 \
  > /tmp/serve2.log 2>&1 &
```

---

## 2. PP=4 部署

### 2.1 启动 Primary / Exporter

```bash
REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_pp4
MODEL=/model/models/Qwen3-8B-FP8

rm -rf ${REG_DIR}/*
mkdir -p ${REG_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm export-weights ${MODEL} \
  --pipeline-parallel-size 4 \
  --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
  --enforce-eager \
  --max-model-len 2048 \
  > /tmp/pp4_export.log 2>&1 &

# 校验：4 个 PP rank 都要有 primary_ready.signal
grep "export complete" /tmp/pp4_export.log
find ${REG_DIR} -type f | sort
```

### 2.2 显存计算

| GPU | 角色 | Primary 占用 | 建议 Secondary util |
|-----|------|-------------|-------------------|
| 0 | PP0 + embedding | ~3.8 GB | 0.78 |
| 1 | PP1 (decoder) | ~2.2 GB | 0.88 |
| 2 | PP2 (decoder) | ~2.2 GB | 0.88 |
| 3 | PP3 + lm_head | ~3.4 GB | 0.83 |

取最紧张的 GPU0，**`--gpu-memory-utilization 0.78`**。

### 2.3 启动 Secondary / Importer

```bash
REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_pp4
MODEL=/model/models/Qwen3-8B-FP8

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${MODEL} \
  --host 0.0.0.0 --port 8000 \
  --served-model-name Qwen3-8B-FP8 \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
  --gpu-memory-utilization 0.78 \
  --max-model-len 2048 \
  > /tmp/pp4_serve.log 2>&1 &

# 校验
grep "IPC-imported" /tmp/pp4_serve.log
grep "Application startup complete" /tmp/pp4_serve.log
curl -s http://127.0.0.1:8000/v1/models
```

---

## 3. 校验命令速查

```bash
# Primary 导出成功
grep "export complete" /tmp/*export.log
grep "ready" /tmp/*export.log

# Registry 完整性（PP=N 应有 N 个 signal）
find ${REG_DIR} -name "primary_ready.signal" | wc -l

# Secondary 权重导入
grep "IPC-imported" /tmp/*serve.log

# Secondary 服务就绪
grep "Application startup complete" /tmp/*serve.log

# 服务验证
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/health

# 推理测试
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-FP8","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

---

## 4. 关键注意事项

1. **JSON 不能转义** — 单引号 `'...'` 内是字面量，不要加 `\"`，变量用 `'"${VAR}"'` 拼接
2. **PP>1 必须加 `--pipeline-parallel-size N`**，否则只导出 rank 0
3. **`--security-opt seccomp=unconfined`** — 解决容器内 `pthread_create` 被拦截
4. **`OPENBLAS_NUM_THREADS=1`** — 防止 OpenBLAS 线程风暴
5. **Primary 和 Secondary 必须在同一物理 GPU**（CUDA IPC 限制）
6. **Primary 用 `export-weights`**（不启动推理，不占 KV cache）
7. **Secondary 不要加 `--enforce-eager`**（保留 CUDA graph）

---

## 5. 原理简图

```
Primary (export-weights, 无 NCCL group, 无 KV cache)
├── 加载模型 + 量化重排
├── reduce_tensor(param.data) → IPC handle
├── 写入 registry（weight_registry.json + primary_ready.signal）
└── 保活，显存仅 ~3-9 GB（纯权重）

Secondary (vllm serve)
├── CPU init 模型结构（0 GPU 显存峰值）
├── 非共享参数 (embed/head/norm) → CUDA
├── 共享参数 → 1-element CPU 占位符
├── 读 registry → IPC handle → param.data = ipc_tensor（零拷贝）
├── CPU buffer 迁移 → CUDA
└── 正常推理（KV cache + CUDA graph）
```
