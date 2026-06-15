#!/bin/bash
# 04_kvcached.sh — kvcached 弹性池化端到端冒烟
# 流程：ENABLE_KVCACHED=1 起 serve（经 sardeenz vllm-entrypoint，含 /kvcached/trim 路由）
#       → 推理 → POST /kvcached/trim 可用 → 日志无 patch 异常栈
# 说明：本脚本不使用 weight-sharing，单纯验证 kvcached monkey-patch 在 0.22.1 能 attach。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

GPU="${GPU:-0}"
PORT="${PORT:-8010}"
# 04 不开 weight-sharing，全卡可用；但若与其他业务/其它 secondary 共卡仍需小心。
UTIL="${UTIL:-0.60}"
SRV_LOG="${LOG_DIR}/04_kvcached_${PORT}.log"
# sardeenz 的 entrypoint（显式 bootstrap kvcached autopatch + /kvcached/trim 路由）
ENTRY="${VLLM_ENTRYPOINT:-/app/apps/backend/scripts/vllm-entrypoint.py}"
[ -f "${ENTRY}" ] || ENTRY="${HERE}/../sardeenz/apps/backend/scripts/vllm-entrypoint.py"

info "=== 04 kvcached (GPU=${GPU}, port=${PORT}) ==="

# 经 sardeenz entrypoint 启动，复刻生产链路；若不存在则退回 vllm serve（仅靠 .pth autopatch）
if [ -f "${ENTRY}" ]; then
    LAUNCH=(python3 "${ENTRY}" serve)
    info "使用 sardeenz entrypoint: ${ENTRY}"
else
    LAUNCH=(vllm serve)
    _c_ylw "[WARN] 未找到 vllm-entrypoint.py，退回 vllm serve（/kvcached/trim 可能不可用）"
fi

ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
KVCACHED_IPC_NAME="kvcached_smoke_gpu${GPU}" \
CUDA_VISIBLE_DEVICES="${GPU}" bg_run "${SRV_LOG}" \
    "${LAUNCH[@]}" "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" --served-model-name "${SERVED_NAME}" \
    --gpu-memory-utilization "${UTIL}" --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager >/dev/null

info "等待 serve 就绪..."
if ! wait_for_log "${SRV_LOG}" "Application startup complete" "${READY_TIMEOUT}"; then
    log_tail "${SRV_LOG}" 60; die "serve 未就绪"
fi

# 关键：日志中不能有 kvcached patch 失败的异常栈
if grep -qiE "Failed to initialize kvcached|patch.*fail|Traceback.*kvcached|kvcached.*Error" "${SRV_LOG}"; then
    log_tail "${SRV_LOG}" 60; die "检测到 kvcached patch 异常"
fi
pass "serve 启动且无 kvcached patch 异常"

# 推理
wait_http "${PORT}" || die "/health 不可用"
assert_curl_chat "${PORT}" || die "chat 推理失败"

# /kvcached/trim 路由
trim=$(curl -fsS -X POST "http://127.0.0.1:${PORT}/kvcached/trim" 2>/dev/null)
if printf '%s' "${trim}" | grep -q '"ok"'; then
    pass "/kvcached/trim 可用: ${trim:0:120}"
else
    _c_ylw "[WARN] /kvcached/trim 未返回 ok（响应: ${trim:0:120}）— 若使用 vllm serve 退回模式属预期"
fi

pass "=== 04 kvcached PASS ==="
