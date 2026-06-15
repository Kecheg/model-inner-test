#!/bin/bash
# run_all.sh — 端到端冒烟测试总入口
# 按序执行 00→05；约定退出码：0=PASS, 77=SKIP, 其它=FAIL。
# 任一 FAIL 即停（除非 CONTINUE_ON_FAIL=1）。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 可用 ONLY="00 01" 只跑指定用例
ONLY="${ONLY:-}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"

ALL=(
    "00_env_check.sh"
    "01_weight_share_1gpu.sh"
    "02_weight_share_pp4.sh"
    "03_dual_secondary.sh"
    "04_kvcached.sh"
    "05_sardeenz_e2e.sh"
    "06_kvcached_with_ws.sh"
    "07_weight_share_tp2.sh"
)

declare -a RESULTS
overall=0

for t in "${ALL[@]}"; do
    tag="${t%%_*}"
    if [ -n "${ONLY}" ] && ! grep -qw "${tag}" <<<"${ONLY}"; then
        continue
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "▶ ${t}"
    echo "════════════════════════════════════════════════════════════"
    bash "${HERE}/${t}"
    rc=$?
    case "${rc}" in
        0)  RESULTS+=("PASS  ${t}");;
        77) RESULTS+=("SKIP  ${t}");;
        *)  RESULTS+=("FAIL  ${t} (rc=${rc})"); overall=1;;
    esac
    if [ "${rc}" -ne 0 ] && [ "${rc}" -ne 77 ] && [ "${CONTINUE_ON_FAIL}" != "1" ]; then
        echo ""
        echo "✗ ${t} 失败，终止（设 CONTINUE_ON_FAIL=1 可继续）"
        break
    fi
done

echo ""
echo "════════════════════════ 汇总 ════════════════════════"
for r in "${RESULTS[@]}"; do
    case "${r}" in
        PASS*) printf '\033[32m%s\033[0m\n' "${r}";;
        SKIP*) printf '\033[33m%s\033[0m\n' "${r}";;
        FAIL*) printf '\033[31m%s\033[0m\n' "${r}";;
    esac
done
echo "══════════════════════════════════════════════════════"
exit "${overall}"
