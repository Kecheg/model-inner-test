#!/bin/bash
# 单卡 Primary / Exporter 启动脚本
# 加载模型，导出权重 IPC handle，保持常驻

set -e

REG_DIR=/workspace/fxp/vllm-weight-share/registry_qwen3_1gpu
MODEL=/model/models/Qwen3-8B-FP8

rm -rf ${REG_DIR}/*
mkdir -p ${REG_DIR}

CUDA_VISIBLE_DEVICES=0 vllm export-weights ${MODEL} \
  --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
  --enforce-eager \
  --max-model-len 2048 \
  > /tmp/export.log 2>&1 &

PID=$!
echo "Primary export-weights started (PID=$PID)"
echo "Log: /tmp/export.log"

# 等待完成
sleep 15
if grep -q "export complete" /tmp/export.log; then
    echo "SUCCESS: Weight export complete"
    echo "Registry: ${REG_DIR}"
    find ${REG_DIR} -type f | sort
else
    echo "FAILED: check /tmp/export.log"
    tail -20 /tmp/export.log
fi
