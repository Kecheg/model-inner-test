#!/bin/bash
# 01_weight_share_1gpu.sh — 单卡权重共享端到端冒烟
# 流程：export-weights(primary) → 校验 registry/signal → serve(secondary)
#       → 校验 IPC-imported + startup → /v1/models + chat 推理
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
# 注意：secondary 和 primary 共享同一张物理 GPU，primary 已占 ~9-10GB。
# 0.22.1 的 request_memory 检查 free >= util*total（比 0.18 更严格），
# 24GB 卡上 util 必须 ≤ (free - 余量) / total ≈ 0.50。
UTIL="${UTIL:-0.50}"
REG_DIR="${REG_ROOT}/qwen3_1gpu"
EXP_LOG="${LOG_DIR}/01_export.log"
SRV_LOG="${LOG_DIR}/01_serve_${PORT}.log"

info "=== 01 单卡权重共享 (GPU=${GPU}, port=${PORT}) ==="
rm -rf "${REG_DIR}"; mkdir -p "${REG_DIR}"

# 1) Primary 导出
CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${EXP_LOG}" \
    vllm export-weights "${MODEL}" \
    --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
    --enforce-eager --max-model-len "${MAX_MODEL_LEN}" >/dev/null

info "等待 primary 导出完成..."
if ! wait_for_log "${EXP_LOG}" "export complete|exported IPC handles|primary_ready" "${READY_TIMEOUT}"; then
    log_tail "${EXP_LOG}"; die "primary 导出未完成"
fi
# 校验 registry 与 signal
sigs=$(find "${REG_DIR}" -name "primary_ready.signal" | wc -l)
regs=$(find "${REG_DIR}" -name "*registry*.json" | wc -l)
[ "${sigs}" -ge 1 ] || { log_tail "${EXP_LOG}"; die "未生成 primary_ready.signal"; }
[ "${regs}" -ge 1 ] || { log_tail "${EXP_LOG}"; die "未生成 weight_registry.json"; }
pass "primary 导出完成 (signal=${sigs}, registry=${regs})"

# 2) Secondary serve
CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${SRV_LOG}" \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" --served-model-name "${SERVED_NAME}" \
    --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
    --gpu-memory-utilization "${UTIL}" --max-model-len "${MAX_MODEL_LEN}" >/dev/null

info "等待 secondary 就绪..."
if ! wait_for_log "${SRV_LOG}" "Application startup complete" "${READY_TIMEOUT}"; then
    log_tail "${SRV_LOG}"; die "secondary 未就绪"
fi
if grep -qE "IPC-imported" "${SRV_LOG}"; then
    pass "secondary 完成 IPC 权重导入"
else
    log_tail "${SRV_LOG}"; die "未检测到 IPC-imported（权重共享未生效）"
fi

# 3) HTTP + 推理断言
wait_http "${PORT}" || die "/health 不可用"
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null || die "/v1/models 失败"
pass "/v1/models 可用"
assert_curl_chat "${PORT}" || die "chat 推理失败"

pass "=== 01 单卡权重共享 PASS ==="
