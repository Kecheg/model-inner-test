#!/bin/bash
# 02_weight_share_pp4.sh — PP=4 权重共享端到端冒烟
# 流程：export-weights(pp4) → 校验 4 个 primary_ready.signal → serve(pp4) → 推理
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

GPUS="${GPUS:-0,1,2,3}"
PP="${PP:-4}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.78}"
REG_DIR="${REG_ROOT}/qwen3_pp${PP}"
EXP_LOG="${LOG_DIR}/02_export_pp${PP}.log"
SRV_LOG="${LOG_DIR}/02_serve_pp${PP}.log"

# 需要至少 PP 张 GPU
ngpu=$(nvidia-smi -L | wc -l)
[ "${ngpu}" -ge "${PP}" ] || { _c_ylw "[SKIP] GPU 数 ${ngpu} < PP ${PP}"; exit 77; }

info "=== 02 PP=${PP} 权重共享 (GPUS=${GPUS}, port=${PORT}) ==="
rm -rf "${REG_DIR}"; mkdir -p "${REG_DIR}"

# 1) Primary 导出（pp4）
CUDA_VISIBLE_DEVICES="${GPUS}" bg_run "${EXP_LOG}" \
    vllm export-weights "${MODEL}" \
    --pipeline-parallel-size "${PP}" \
    --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
    --enforce-eager --max-model-len "${MAX_MODEL_LEN}" >/dev/null

info "等待 pp${PP} 导出完成..."
if ! wait_for_log "${EXP_LOG}" "export complete|exported IPC handles" "${READY_TIMEOUT}"; then
    log_tail "${EXP_LOG}"; die "pp${PP} 导出未完成"
fi
sigs=$(find "${REG_DIR}" -name "primary_ready.signal" | wc -l)
[ "${sigs}" -ge "${PP}" ] || { log_tail "${EXP_LOG}"; die "primary_ready.signal 数=${sigs}，应为 ${PP}（PP rank 未全部导出）"; }
pass "pp${PP} 导出完成，${sigs} 个 rank signal 齐全"

# 2) Secondary serve（pp4）
CUDA_VISIBLE_DEVICES="${GPUS}" bg_run "${SRV_LOG}" \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" --served-model-name "${SERVED_NAME}" \
    --pipeline-parallel-size "${PP}" --tensor-parallel-size 1 \
    --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
    --gpu-memory-utilization "${UTIL}" --max-model-len "${MAX_MODEL_LEN}" >/dev/null

info "等待 secondary(pp${PP}) 就绪..."
if ! wait_for_log "${SRV_LOG}" "Application startup complete" "${READY_TIMEOUT}"; then
    log_tail "${SRV_LOG}"; die "secondary(pp${PP}) 未就绪"
fi
grep -qE "IPC-imported" "${SRV_LOG}" || { log_tail "${SRV_LOG}"; die "未检测到 IPC-imported"; }
pass "secondary(pp${PP}) IPC 导入完成"

# 3) 推理
wait_http "${PORT}" || die "/health 不可用"
assert_curl_chat "${PORT}" || die "chat 推理失败"

pass "=== 02 PP=${PP} 权重共享 PASS ==="
