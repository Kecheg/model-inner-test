#!/bin/bash
# tests/lib/common.sh
# 端到端冒烟测试公共函数库。所有冒烟脚本 source 本文件。
#
# 约定：
#   - 全部设计为「镜像内 + GPU」运行（宿主机无 torch/vllm）。
#   - 参数走环境变量，带默认值，便于 CI / 手工覆盖。

set -uo pipefail

# ───────────────────────── 全局可配置项 ─────────────────────────
export MODEL="${MODEL:-/model/models/Qwen3-8B-FP8}"
export SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL}")}"
export REG_ROOT="${REG_ROOT:-/tmp/ws-registry}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export LOG_DIR="${LOG_DIR:-/tmp/ws-smoke-logs}"
export READY_TIMEOUT="${READY_TIMEOUT:-300}"   # 秒：等待 serve / export 就绪
export HTTP_TIMEOUT="${HTTP_TIMEOUT:-120}"     # 秒：等待端口可访问
mkdir -p "${LOG_DIR}"

# 记录本次脚本启动的后台进程组 ID（PGID）。
# 我们用 setsid 启动每个后台进程，让它独占进程组；cleanup 时
# `kill -- -PGID` 一次性收掉整组（含子进程，例如 vllm 的 EngineCore）。
# 这避免了「pkill -f vllm」误杀宿主上不相关进程（尤其 --pid=host 时）。
_SMOKE_PGIDS=()

# ───────────────────────── 颜色 / 日志 ─────────────────────────
_c_red()   { printf '\033[31m%s\033[0m\n' "$*"; }
_c_grn()   { printf '\033[32m%s\033[0m\n' "$*"; }
_c_ylw()   { printf '\033[33m%s\033[0m\n' "$*"; }

info()  { _c_ylw "[*] $*"; }
pass()  { _c_grn "[PASS] $*"; }
fail()  { _c_red "[FAIL] $*"; }

# 致命错误：打印消息 + 退出非 0（trap 会做清理）
die() { fail "$*"; exit 1; }

# ───────────────────────── 进程管理 ─────────────────────────
# 后台启动一条命令；用 setsid + bash exec 让 cmd 成为新进程组组长，
# 并把真实 PID（= PGID）写到临时文件回读。
# 注意：直接 setsid cmd & 的 $! 会因为 setsid 自身 fork 而漂移到错的 PID，
# 因此必须让 bash 自己用 exec 替换为 cmd 以保持 PID 稳定。
# 用法：bg_run <logfile> <cmd...>
bg_run() {
    local logf="$1"; shift
    info "启动: $* (日志: ${logf})"
    local pidf
    pidf=$(mktemp)
    setsid bash -c 'echo $$ > "$0"; shift; exec "$@"' "${pidf}" _ "$@" \
        > "${logf}" 2>&1 < /dev/null &
    # 等 PID 文件写入
    local i=0
    while [ "${i}" -lt 20 ] && [ ! -s "${pidf}" ]; do
        sleep 0.1; i=$((i+1))
    done
    local real_pid
    real_pid=$(cat "${pidf}" 2>/dev/null)
    rm -f "${pidf}"
    if [ -z "${real_pid}" ]; then
        fail "bg_run 未能获取真实 PID（命令: $*）"
        return 1
    fi
    _SMOKE_PGIDS+=("${real_pid}")
    echo "${real_pid}"
}

# 只杀本脚本启动的进程组——不触及任何宿主上不相关的进程。
# 即使在 --pid=host 模式下也是安全的。
cleanup_procs() {
    local pgid
    for pgid in "${_SMOKE_PGIDS[@]:-}"; do
        [ -n "${pgid}" ] || continue
        # 先 TERM 整组
        kill -TERM -- "-${pgid}" 2>/dev/null
    done
    # 给点时间正常退出
    sleep 2
    for pgid in "${_SMOKE_PGIDS[@]:-}"; do
        [ -n "${pgid}" ] || continue
        # 仍存活则 KILL
        kill -KILL -- "-${pgid}" 2>/dev/null
    done
}

# 注册退出清理
trap cleanup_procs EXIT INT TERM

# ───────────────────────── 等待 / 断言 ─────────────────────────
# 等待日志文件出现某个 pattern。用法：wait_for_log <logfile> <pattern> [timeout]
wait_for_log() {
    local logf="$1" pat="$2" to="${3:-${READY_TIMEOUT}}" i=0
    while [ "${i}" -lt "${to}" ]; do
        if grep -qE "${pat}" "${logf}" 2>/dev/null; then return 0; fi
        # 若进程已死，提前失败
        sleep 1; i=$((i+1))
    done
    return 1
}

# 等待 HTTP 端口的 /health 返回 200。用法：wait_http <port> [timeout]
wait_http() {
    local port="$1" to="${2:-${HTTP_TIMEOUT}}" i=0
    while [ "${i}" -lt "${to}" ]; do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then return 0; fi
        sleep 1; i=$((i+1))
    done
    return 1
}

# 对 chat/completions 发请求并断言返回非空内容。用法：assert_curl_chat <port> [model_name]
assert_curl_chat() {
    local port="$1" model="${2:-${SERVED_NAME}}"
    local resp
    resp=$(curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${model}"'","messages":[{"role":"user","content":"你好，用一句话自我介绍"}],"max_tokens":32}' 2>/dev/null)
    if [ -z "${resp}" ]; then
        fail "chat 请求无响应 (port=${port})"; return 1
    fi
    # 提取 content 字段，要求非空
    local content
    content=$(printf '%s' "${resp}" | python3 -c \
        'import sys,json;
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception as e:
    print("")' 2>/dev/null)
    if [ -z "${content// /}" ]; then
        fail "chat 返回内容为空 (port=${port})。原始响应: ${resp:0:300}"; return 1
    fi
    pass "chat 推理返回 (port=${port}): ${content:0:80}"
    return 0
}

# 打印日志尾部，用于失败排查。用法：log_tail <logfile> [n]
log_tail() {
    local logf="$1" n="${2:-40}"
    echo "----- tail -n ${n} ${logf} -----"
    tail -n "${n}" "${logf}" 2>/dev/null || echo "(无日志)"
    echo "--------------------------------"
}

# 校验模型目录存在，否则跳过（exit 0 不算失败，仅 SKIP）
require_model() {
    if [ ! -d "${MODEL}" ]; then
        _c_ylw "[SKIP] 模型目录不存在: ${MODEL} （设置 MODEL=... 指向有效模型后重跑）"
        exit 77   # 77 = 约定的 SKIP 码
    fi
}

# 校验在容器内 / 有 GPU
require_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || die "未找到 nvidia-smi（需在 GPU 容器内运行）"
    nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi 无法列出 GPU"
}

require_vllm() {
    python3 -c "import vllm" 2>/dev/null || die "当前环境未安装 vllm（请在构建好的镜像内运行）"
}
