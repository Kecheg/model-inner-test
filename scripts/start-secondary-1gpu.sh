#!/bin/bash
# 单卡 Secondary / Importer 启动脚本
# 从 Primary 共享权重，启动推理服务

set -e

REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_1gpu
MODEL=/model/models/Qwen3-8B-FP8
PORT=${1:-8000}
UTIL=${2:-0.60}

CUDA_VISIBLE_DEVICES=0 vllm serve ${MODEL} \
  --host 0.0.0.0 --port ${PORT} \
  --served-model-name Qwen3-8B-FP8 \
  --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
  --gpu-memory-utilization ${UTIL} \
  --max-model-len 2048 \
  > /tmp/serve-${PORT}.log 2>&1 &

PID=$!
echo "Secondary started (PID=$PID, port=${PORT}, util=${UTIL})"
echo "Log: /tmp/serve-${PORT}.log"

# 等待就绪
echo "Waiting for startup..."
for i in $(seq 1 30); do
    if grep -q "Application startup complete" /tmp/serve-${PORT}.log 2>/dev/null; then
        echo "SUCCESS: Server ready on port ${PORT}"
        grep "IPC-imported" /tmp/serve-${PORT}.log
        curl -s http://127.0.0.1:${PORT}/v1/models
        exit 0
    fi
    sleep 2
done
echo "FAILED: check /tmp/serve-${PORT}.log"
tail -20 /tmp/serve-${PORT}.log
