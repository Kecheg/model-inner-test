#!/bin/bash
# 03_dual_secondary.sh — 同 GPU 双 secondary 共享同一 primary
# 流程：export(primary) → secondary#1(8000) + secondary#2(8001) → 两端都能推理
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

GPU="${GPU:-0}"
PORT1="${PORT1:-8000}"
PORT2="${PORT2:-8001}"
# 单 GPU(24G) 关键内存账（0.22.1 secondary 模式）：
#   primary 占 ~9.6GB → 剩 14.4GB（secondary#1 启动）→ 剩 ~11.5GB（secondary#2 启动）
#   non_kv_cache_memory ≈ 8.8 GiB（自有 embed/lm_head 2.33 + IPC 共享权重 6.47 + activations）
#   available_kv_cache = util*total - non_kv_cache_memory，需要 > 0
# 故 util 必须 ≥ ~0.40 才有正的 KV cache。
UTIL1="${UTIL1:-0.50}"
UTIL2="${UTIL2:-0.42}"
# secondary#2 用更短的 max_model_len 压低单序列 KV 占用
MAXLEN2="${MAXLEN2:-1024}"
REG_DIR="${REG_ROOT}/qwen3_dual"
EXP_LOG="${LOG_DIR}/03_export.log"
S1_LOG="${LOG_DIR}/03_serve_${PORT1}.log"
S2_LOG="${LOG_DIR}/03_serve_${PORT2}.log"

info "=== 03 同 GPU 双 secondary (GPU=${GPU}, ports=${PORT1}/${PORT2}) ==="
rm -rf "${REG_DIR}"; mkdir -p "${REG_DIR}"

# 1) Primary
CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${EXP_LOG}" \
    vllm export-weights "${MODEL}" \
    --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
    --enforce-eager --max-model-len "${MAX_MODEL_LEN}" >/dev/null
wait_for_log "${EXP_LOG}" "export complete|exported IPC handles|primary_ready" "${READY_TIMEOUT}" \
    || { log_tail "${EXP_LOG}"; die "primary 导出未完成"; }
pass "primary 导出完成"

# 2) 两个 secondary
start_secondary() {
    local port="$1" util="$2" logf="$3" maxlen="$4"
    CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${logf}" \
        vllm serve "${MODEL}" \
        --host 0.0.0.0 --port "${port}" --served-model-name "${SERVED_NAME}" \
        --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
        --gpu-memory-utilization "${util}" --max-model-len "${maxlen}" >/dev/null
}

start_secondary "${PORT1}" "${UTIL1}" "${S1_LOG}" "${MAX_MODEL_LEN}"
start_secondary "${PORT2}" "${UTIL2}" "${S2_LOG}" "${MAXLEN2}"

for pair in "${PORT1}:${S1_LOG}" "${PORT2}:${S2_LOG}"; do
    port="${pair%%:*}"; logf="${pair##*:}"
    info "等待 secondary(${port}) 就绪..."
    wait_for_log "${logf}" "Application startup complete" "${READY_TIMEOUT}" \
        || { log_tail "${logf}"; die "secondary(${port}) 未就绪"; }
    grep -qE "IPC-imported" "${logf}" || { log_tail "${logf}"; die "secondary(${port}) 未 IPC 导入"; }
    wait_http "${port}" || die "secondary(${port}) /health 不可用"
    assert_curl_chat "${port}" || die "secondary(${port}) 推理失败"
done

pass "=== 03 双 secondary 共享 PASS ==="
