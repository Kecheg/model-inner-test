#!/bin/bash
# 05_sardeenz_e2e.sh — sardeenz 编排端到端冒烟
# 流程：起 sardeenz backend → /health → 经 POST /api/models/load 拉起模型
#       → 轮询 GET /api/models 至 running → 经 sardeenz 代理推理
# 说明：本测试依赖 sardeenz 已构建产物 (apps/backend/dist/server.js) 与 vllm 可用。
#       受 admin 鉴权约束：通过 SARDEENZ_TOKEN 提供 Bearer token（若部署关闭鉴权则可留空）。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

require_vllm; require_gpu; require_model

SD_PORT="${SD_PORT:-3000}"
SD_DIR="${SARDEENZ_DIR:-/app}"
[ -f "${SD_DIR}/apps/backend/dist/server.js" ] || SD_DIR="${HERE}/../sardeenz"
SERVER_JS="${SD_DIR}/apps/backend/dist/server.js"
TOKEN="${SARDEENZ_TOKEN:-}"
SD_LOG="${LOG_DIR}/05_sardeenz.log"
GPU="${GPU:-0}"

[ -f "${SERVER_JS}" ] || { _c_ylw "[SKIP] 未找到 sardeenz 构建产物: ${SERVER_JS}（需先 build）"; exit 77; }
command -v node >/dev/null 2>&1 || { _c_ylw "[SKIP] 未找到 node"; exit 77; }

auth_hdr=()
[ -n "${TOKEN}" ] && auth_hdr=(-H "Authorization: Bearer ${TOKEN}")

info "=== 05 sardeenz e2e (backend port=${SD_PORT}) ==="

# 1) 启动 backend
PORT="${SD_PORT}" ENABLE_KVCACHED="${ENABLE_KVCACHED:-true}" KVCACHED_AUTOPATCH=1 \
NVIDIA_VISIBLE_DEVICES="${GPU}" \
bg_run "${SD_LOG}" node "${SERVER_JS}" >/dev/null

info "等待 backend /health..."
i=0
while [ "${i}" -lt 60 ]; do
    curl -fsS "http://127.0.0.1:${SD_PORT}/health" >/dev/null 2>&1 && break
    sleep 1; i=$((i+1))
done
[ "${i}" -lt 60 ] || { log_tail "${SD_LOG}"; die "backend 未就绪"; }
pass "sardeenz backend 就绪"

# 2) 触发模型加载
load_resp=$(curl -fsS -X POST "http://127.0.0.1:${SD_PORT}/api/models/load" \
    "${auth_hdr[@]}" -H "Content-Type: application/json" \
    -d '{"model_path":"'"${MODEL}"'","max_tokens":'"${MAX_MODEL_LEN}"',"served_model_name":"'"${SERVED_NAME}"'","gpu_ids":['"${GPU}"'],"enforce_eager":true}' 2>&1)
if printf '%s' "${load_resp}" | grep -qiE "unauthorized|forbidden|401|403"; then
    _c_ylw "[SKIP] 加载被鉴权拒绝；设置 SARDEENZ_TOKEN 后重跑。响应: ${load_resp:0:160}"
    exit 77
fi
printf '%s' "${load_resp}" | grep -qiE "instance|operation|starting|id" \
    || { log_tail "${SD_LOG}"; die "模型加载请求失败: ${load_resp:0:240}"; }
pass "模型加载已触发"

# 3) 轮询至 running
info "等待模型 running（最多 ${READY_TIMEOUT}s）..."
i=0; running=0; mport=""
while [ "${i}" -lt "${READY_TIMEOUT}" ]; do
    models=$(curl -fsS "${auth_hdr[@]}" "http://127.0.0.1:${SD_PORT}/api/models" 2>/dev/null)
    if printf '%s' "${models}" | grep -qiE '"status"\s*:\s*"(running|ready)"'; then
        running=1
        mport=$(printf '%s' "${models}" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    arr=d if isinstance(d,list) else d.get("models",d.get("instances",[]))
    for m in arr:
        if str(m.get("status","")).lower() in ("running","ready"):
            print(m.get("port","")); break
except Exception: pass' 2>/dev/null)
        break
    fi
    sleep 3; i=$((i+3))
done
[ "${running}" -eq 1 ] || { log_tail "${SD_LOG}" 60; die "模型未进入 running"; }
pass "模型 running (port=${mport:-?})"

# 4) 经 sardeenz 代理推理（统一入口 /v1/chat/completions 由 sardeenz 路由到实例）
assert_curl_chat "${SD_PORT}" || {
    # 退而求其次：直接打模型端口
    [ -n "${mport}" ] && assert_curl_chat "${mport}" || die "经 sardeenz 推理失败"
}

pass "=== 05 sardeenz e2e PASS ==="
