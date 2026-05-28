# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""参数分类：决定哪些参数可以跨进程共享，哪些必须每个进程独立持有。

核心规则：
  - Transformer 解码器层（model.layers.N.*）→ 可共享
  - 嵌入层（embed_tokens）、输出头（lm_head）、最终归一化（norm）→ 不共享
  - Attention KV cache 量化 scale（q/k/v/prob_scale）→ 不共享（每进程独立）
"""

from collections.abc import Set as AbstractSet

# ----------------------------------------------------------------
# 可共享参数前缀：解码器层的权重，跨主流模型架构
# ----------------------------------------------------------------
SHAREABLE_PARAM_PREFIXES: tuple[str, ...] = (
    "model.layers.",           # Llama, Qwen2, Mistral, Mixtral, Gemma 等主流架构
    "transformer.h.",          # Falcon, GPT-J, BLOOM
    "transformer.blocks.",     # MPT
    "transformer.layers.",     # OPT
    "gpt_neox.layers.",        # GPT-NeoX, Pythia
    "decoder.layers.",         # T5（encoder 权重不共享）
    "language_model.model.layers.",  # Phi-3, ChatGLM 等
    "encoder.layer.",          # BERT, RoBERTa, XLM-RoBERTa, BGE 等（原生参数名）
    "model.encoder.layer.",    # BERT/RoBERTa 等（vllm 加载后添加 model. 前缀）
    "roberta.encoder.layer.",  # XLMRobertaForSequenceClassification / reranker
)

# ----------------------------------------------------------------
# 不可共享参数——精确名称列表（优先级最高，O(1) 查找）
# ----------------------------------------------------------------
NON_SHAREABLE_PARAM_NAMES: AbstractSet[str] = frozenset({
    # 词嵌入和 lm_head 在推理时只读，可以 IPC 共享
    "transformer.wte.weight",            # GPT-2 / GPT-J 词嵌入
    "transformer.ln_f.weight",           # GPT-2 / Falcon 最终 LayerNorm
    "transformer.ln_f.bias",
    "transformer.word_embeddings.weight",# OPT 词嵌入
    "gpt_neox.embed_in.weight",          # Pythia / GPT-NeoX 词嵌入
    "gpt_neox.final_layer_norm.weight",  # Pythia / GPT-NeoX 最终 LayerNorm
    "gpt_neox.final_layer_norm.bias",
    "embed_out.weight",                  # GPT-NeoX lm_head 变体
    "embed_in.weight",                   # 另一个变体
    "language_model.embed_tokens.weight",# Phi-3, ChatGLM
})

# ----------------------------------------------------------------
# 不可共享参数——前缀列表（覆盖子参数，如 scales/bias）
# ----------------------------------------------------------------
NON_SHAREABLE_PARAM_PREFIXES: tuple[str, ...] = (
    "model.vision_",         # 多模态视觉编码器
    "model.mm_projector",    # 多模态投影层
    "model.embed_tokens.",     # Llama / Qwen 词嵌入
    "model.norm.",             # Llama / Qwen 最终 RMSNorm / LayerNorm
    "lm_head.",                # 输出投影
    "transformer.wte.",
    "transformer.ln_f.",
    "transformer.word_embeddings.",
    "gpt_neox.embed_in.",
    "gpt_neox.final_layer_norm.",
    "language_model.embed_tokens.",
    "embed_out.",
    "embed_in.",
    "embeddings.",             # BERT / RoBERTa 原生参数名
    "model.embeddings.",       # vllm 加载后添加 model. 前缀
    "roberta.embeddings.",     # XLMRobertaForSequenceClassification / reranker
    "pooler.",                 # BERT / RoBERTa 原生参数名
    "model.pooler.",           # vllm 加载后添加 model. 前缀
    "roberta.pooler.",         # XLMRobertaForSequenceClassification / reranker
    "score.",                  # SequenceClassification / reranker 分类头
    "classifier.",             # SequenceClassification 分类头变体
)


def is_shareable_parameter(param_name: str) -> bool:
    """判断一个参数是否可以跨进程共享。

    三层过滤逻辑（按顺序）：
    1. 精确名称排除：O(1) frozenset 查找，快速排除已知不共享参数
    2. 前缀排除：排除嵌入/头/norm 的所有子参数（含 scale、bias 等）
    3. 后缀排除：排除 Attention 量化 scale（虽在解码器层内，但每进程独立）
    4. 前缀匹配：在解码器层内 → 可共享
    """
    # 第一层：精确名称排除（最快路径）
    if param_name in NON_SHAREABLE_PARAM_NAMES:
        return False

    # 第二层：前缀排除（嵌入/头/norm 下的所有参数）
    for prefix in NON_SHAREABLE_PARAM_PREFIXES:
        if param_name.startswith(prefix):
            return False

    # 第三层：后缀排除——Attention KV cache 量化 scale
    # 这些参数物理上位于 model.layers.N.self_attn.*，但每进程 KV cache 独立，
    # 不能共享，否则会影响 KV cache 量化精度
    _NON_SHAREABLE_SUFFIXES = (
        ".q_scale", ".k_scale", ".v_scale", ".prob_scale",
    )
    for suffix in _NON_SHAREABLE_SUFFIXES:
        if param_name.endswith(suffix):
            return False

    # 第四层：在解码器层内 → 可共享
    for prefix in SHAREABLE_PARAM_PREFIXES:
        if param_name.startswith(prefix):
            return True

    return False


def get_shareable_param_names(model) -> set[str]:
    """返回 model 中所有可共享参数的名称集合。"""
    return {n for n, _ in model.named_parameters()
            if is_shareable_parameter(n)}


def get_non_shareable_param_names(model) -> set[str]:
    """返回 model 中所有不可共享参数的名称集合。"""
    return {n for n, _ in model.named_parameters()
            if not is_shareable_parameter(n)}
