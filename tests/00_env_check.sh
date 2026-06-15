#!/bin/bash
# 00_env_check.sh — 环境与 overlay 完整性检查（无需加载模型，秒级）
# 判定：vllm 版本为 0.22.x、weight-sharing overlay 已就位、CLI 子命令已注册、
#       kvcached autopatch 可 import、GPU 可见。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

info "=== 00 环境检查开始 ==="
rc=0

# 1) vllm 可导入 + 版本
require_vllm
VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
info "vllm.__version__ = ${VER}"
case "${VER}" in
    0.22.*) pass "vllm 版本符合预期 (0.22.x)";;
    *) fail "vllm 版本非 0.22.x（实际 ${VER}）"; rc=1;;
esac

# 2) weight-sharing 配置类可导入
if python3 -c "from vllm.config.weight_sharing import WeightSharingConfig" 2>/dev/null; then
    pass "WeightSharingConfig 可导入"
else
    fail "WeightSharingConfig 导入失败（config 改动未生效）"; rc=1
fi

# 3) parameter_filter 行为正确
if python3 -c "
from vllm.distributed.weight_sharing.parameter_filter import is_shareable_parameter as f
assert f('model.layers.0.self_attn.qkv_proj.weight') is True
assert f('model.embed_tokens.weight') is False
assert f('lm_head.weight') is False
print('ok')" 2>/dev/null | grep -q ok; then
    pass "is_shareable_parameter 分类正确"
else
    fail "is_shareable_parameter 行为不符（parameter_filter 迁移有问题）"; rc=1
fi

# 4) CLI 子命令注册
HELP=$(vllm --help 2>&1)
for sub in "export-weights" "export-multi-weights"; do
    if printf '%s' "${HELP}" | grep -q "${sub}"; then
        pass "CLI 子命令已注册: ${sub}"
    else
        fail "CLI 缺少子命令: ${sub}（entrypoints/cli/main.py 未生效）"; rc=1
    fi
done

# 5) kvcached autopatch 可 import（仅 import，不实际 patch 运行引擎）
if python3 -c "import kvcached.integration.vllm.autopatch" 2>/dev/null; then
    pass "kvcached autopatch import 成功"
else
    fail "kvcached autopatch import 失败（Phase 2 未完成时此项预期为 FAIL）"; rc=1
fi

# 6) GPU 可见
require_gpu
pass "GPU 可见: $(nvidia-smi -L | wc -l) 张"

[ "${rc}" -eq 0 ] && pass "=== 00 环境检查全部通过 ===" || fail "=== 00 环境检查存在失败项 ==="
exit "${rc}"
