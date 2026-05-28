#!/bin/bash
# PP=4 Secondary / Importer 启动脚本

set -e

REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_pp4
MODEL=/model/models/Qwen3-8B-FP8
PORT=${1:-8000}
UTIL=${2:-0.78}

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${MODEL} \
  --host 0.0.0.0 --port ${PORT} \
  --served-model-name Qwen3-8B-FP8 \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
  --gpu-memory-utilization ${UTIL} \
  --max-model-len 2048 \
  > /tmp/pp4_serve-${PORT}.log 2>&1 &

PID=$!
echo "Secondary PP=4 started (PID=$PID, port=${PORT}, util=${UTIL})"

for i in $(seq 1 40); do
    if grep -q "Application startup complete" /tmp/pp4_serve-${PORT}.log 2>/dev/null; then
        echo "SUCCESS on port ${PORT}"
        grep "IPC-imported" /tmp/pp4_serve-${PORT}.log
        grep "Available KV cache" /tmp/pp4_serve-${PORT}.log
        curl -s http://127.0.0.1:${PORT}/v1/models
        exit 0
    fi
    sleep 3
done
echo "FAILED"
tail -20 /tmp/pp4_serve-${PORT}.log
