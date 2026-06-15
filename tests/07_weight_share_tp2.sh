#!/bin/bash
# 07_weight_share_tp2.sh — TP=2 weight-sharing 端到端
# 流程：primary export-weights --tensor-parallel-size 2（生成 pp0_tp0 + pp0_tp1 两套 registry）
#       → secondary serve --tensor-parallel-size 2 → 两 rank 各自 IPC import → chat
# 关键：每个 TP rank 独立的 IPC registry 目录、model_hash + tp_rank/tp_size 校验
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.50}"
REG_DIR="${REG_ROOT}/qwen3_tp${TP}"
EXP_LOG="${LOG_DIR}/07_export_tp${TP}.log"
SRV_LOG="${LOG_DIR}/07_serve_tp${TP}.log"

# 需要至少 TP 张 GPU
ngpu=$(nvidia-smi -L | wc -l)
[ "${ngpu}" -ge "${TP}" ] || { _c_ylw "[SKIP] GPU 数 ${ngpu} < TP ${TP}"; exit 77; }

info "=== 07 TP=${TP} weight-sharing (GPUS=${GPUS}, port=${PORT}) ==="
rm -rf "${REG_DIR}"; mkdir -p "${REG_DIR}"

# 1) primary 导出（TP=2）
CUDA_VISIBLE_DEVICES="${GPUS}" bg_run "${EXP_LOG}" \
    vllm export-weights "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --weight-sharing-config '{"mode":"primary","registry_path":"'"${REG_DIR}"'"}' \
    --enforce-eager --max-model-len "${MAX_MODEL_LEN}" >/dev/null

info "等待 TP=${TP} 导出完成..."
if ! wait_for_log "${EXP_LOG}" "export complete|exported IPC handles" "${READY_TIMEOUT}"; then
    log_tail "${EXP_LOG}" 60; die "TP=${TP} 导出未完成"
fi

# 关键校验：TP rank signal 数 == TP（pp0_tp0/signal, pp0_tp1/signal, ...）
sigs=$(find "${REG_DIR}" -name "primary_ready.signal" | wc -l)
[ "${sigs}" -ge "${TP}" ] || { log_tail "${EXP_LOG}" 60; die "primary_ready.signal 数=${sigs}，应为 ${TP}（TP rank 未全部导出）"; }
# 进一步：每个 tp_rank 都应有自己的子目录
tp_dirs=$(find "${REG_DIR}" -type d -name "pp*_tp*" | sort)
echo "${tp_dirs}"
[ "$(echo "${tp_dirs}" | wc -l)" -ge "${TP}" ] || die "TP rank 目录数不足"
pass "TP=${TP} 导出完成，${sigs} 个 rank signal + ${TP} 个 tp 目录齐全"

# 2) secondary 启动（TP=2）
CUDA_VISIBLE_DEVICES="${GPUS}" bg_run "${SRV_LOG}" \
    vllm serve "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" --pipeline-parallel-size 1 \
    --weight-sharing-config '{"mode":"secondary","registry_path":"'"${REG_DIR}"'"}' \
    --gpu-memory-utilization "${UTIL}" --max-model-len "${MAX_MODEL_LEN}" >/dev/null

info "等待 secondary(tp=${TP}) 就绪..."
if ! wait_for_log "${SRV_LOG}" "Application startup complete" "${READY_TIMEOUT}"; then
    log_tail "${SRV_LOG}" 80; die "secondary(tp=${TP}) 未就绪"
fi

# 关键校验：每个 TP rank 都做了 IPC 导入
imported_count=$(grep -cE "IPC-imported" "${SRV_LOG}")
[ "${imported_count}" -ge "${TP}" ] || { log_tail "${SRV_LOG}" 80; die "IPC-imported 次数=${imported_count}，应至少 ${TP}（每 rank 一次）"; }
pass "secondary(tp=${TP}) 两个 rank 都完成 IPC 导入 (出现 ${imported_count} 次)"

# 3) 推理
wait_http "${PORT}" || die "/health 不可用"
assert_curl_chat "${PORT}" || die "chat 推理失败"

pass "=== 07 TP=${TP} weight-sharing PASS ==="
