#!/bin/bash
# PP=4 Primary / Exporter 启动脚本

set -e

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

PID=$!
echo "Primary PP=4 export-weights started (PID=$PID)"
echo "Log: /tmp/pp4_export.log"

sleep 20
if [ $(find ${REG_DIR} -name "primary_ready.signal" | wc -l) -eq 4 ]; then
    echo "SUCCESS: 4 PP ranks exported"
    find ${REG_DIR} -type f | sort
else
    echo "FAILED: check /tmp/pp4_export.log"
    tail -20 /tmp/pp4_export.log
fi
