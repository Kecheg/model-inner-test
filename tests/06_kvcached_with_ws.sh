#!/bin/bash
# 06_kvcached_with_ws.sh — kvcached + weight-sharing 共存（生产真实组合）
# 流程：primary export-weights → secondary 同时启用 kvcached(autopatch) + weight-sharing
#       → 验证日志无任何 patch 异常 / IPC-imported 出现 / chat 成功
# 关键：kvcached 接管 KV cache 分配，weight-sharing 接管模型权重，两者各管各的
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.50}"
REG_DIR="${REG_ROOT}/qwen3_ws_kvc"
EXP_LOG="${LOG_DIR}/06_export.log"
SRV_LOG="${LOG_DIR}/06_serve_${PORT}.log"

info "=== 06 weight-sharing + kvcached 共存 (GPU=${GPU}, port=${PORT}) ==="
rm -rf "${REG_DIR}"; mkdir -p "${REG_DIR}"

# 1) primary 导出（不开 kvcached）
CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${EXP_LOG}" \
    vllm export-weights "${MODEL}" \
    --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
    --enforce-eager --max-model-len "${MAX_MODEL_LEN}" >/dev/null
wait_for_log "${EXP_LOG}" "export complete|exported IPC handles|primary_ready" "${READY_TIMEOUT}" \
    || { log_tail "${EXP_LOG}"; die "primary 未完成"; }
pass "primary 导出完成"

# 2) secondary 同时启用 kvcached + weight-sharing
ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
KVCACHED_IPC_NAME="kvcached_ws_smoke_gpu${GPU}" \
CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${SRV_LOG}" \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" --served-model-name "${SERVED_NAME}" \
    --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
    --gpu-memory-utilization "${UTIL}" --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager >/dev/null

info "等待 secondary 就绪（kvcached + ws 双加成）..."
if ! wait_for_log "${SRV_LOG}" "Application startup complete" "${READY_TIMEOUT}"; then
    log_tail "${SRV_LOG}" 80; die "secondary 未就绪"
fi

# 3) 关键校验：两个 monkey-patch 链路都没异常
if grep -qiE "Failed to initialize kvcached|patch.*fail|Traceback.*kvcached" "${SRV_LOG}"; then
    log_tail "${SRV_LOG}" 80; die "kvcached patch 异常"
fi
grep -qE "IPC-imported" "${SRV_LOG}" || { log_tail "${SRV_LOG}" 80; die "weight-sharing IPC 未生效"; }
pass "kvcached patch 无异常；weight-sharing IPC 导入完成"

# 4) 推理
wait_http "${PORT}" || die "/health 不可用"
assert_curl_chat "${PORT}" || die "chat 推理失败"

pass "=== 06 kvcached + weight-sharing 共存 PASS ==="
