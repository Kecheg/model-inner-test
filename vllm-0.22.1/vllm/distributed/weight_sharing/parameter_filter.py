# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""参数分类：决定哪些参数可以跨进程共享，哪些必须每个进程独立持有。"""

from collections.abc import Iterable

from vllm.config.weight_sharing import WeightSharingConfig

# ----------------------------------------------------------------
# 默认可共享参数前缀：Transformer 主体层权重，跨主流模型架构
# ----------------------------------------------------------------
SHAREABLE_PARAM_PREFIXES: tuple[str, ...] = (
    "model.layers.",
    "transformer.h.",
    "transformer.blocks.",
    "transformer.layers.",
    "gpt_neox.layers.",
    "decoder.layers.",
    "language_model.model.layers.",
    "encoder.layer.",
    "model.encoder.layer.",
    "roberta.encoder.layer.",
)

EMBEDDING_PARAM_PREFIXES: tuple[str, ...] = (
    "model.embed_tokens.",
    "transformer.wte.",
    "transformer.word_embeddings.",
    "gpt_neox.embed_in.",
    "language_model.embed_tokens.",
    "language_model.model.embed_tokens.",
    "embeddings.",
    "model.embeddings.",
    "roberta.embeddings.",
)
EMBEDDING_PARAM_NAMES: frozenset[str] = frozenset({
    "transformer.wte.weight",
    "transformer.word_embeddings.weight",
    "gpt_neox.embed_in.weight",
    "embed_in.weight",
    "language_model.embed_tokens.weight",
})

LM_HEAD_PARAM_PREFIXES: tuple[str, ...] = (
    "lm_head.",
    "language_model.lm_head.",
    "embed_out.",
)
LM_HEAD_PARAM_NAMES: frozenset[str] = frozenset({
    "embed_out.weight",
})

NORM_PARAM_PREFIXES: tuple[str, ...] = (
    "model.norm.",
    "transformer.ln_f.",
    "gpt_neox.final_layer_norm.",
)
NORM_PARAM_NAMES: frozenset[str] = frozenset({
    "transformer.ln_f.weight",
    "transformer.ln_f.bias",
    "gpt_neox.final_layer_norm.weight",
    "gpt_neox.final_layer_norm.bias",
})

VISION_ENCODER_PARAM_PREFIXES: tuple[str, ...] = (
    "model.vision_",
    "visual.",
    "vision_model.",
    "vision_tower.",
    "image_encoder.",
)

PROJECTOR_PARAM_PREFIXES: tuple[str, ...] = (
    "model.mm_projector",
    "mm_projector.",
    "multi_modal_projector.",
)

POOLER_PARAM_PREFIXES: tuple[str, ...] = (
    "pooler.",
    "model.pooler.",
    "roberta.pooler.",
    "score.",
    "classifier.",
)

_NON_SHAREABLE_SUFFIXES: tuple[str, ...] = (
    ".q_scale", ".k_scale", ".v_scale", ".prob_scale",
)


def _config_or_default(config: WeightSharingConfig | None) -> WeightSharingConfig:
    return config if config is not None else WeightSharingConfig()


def _has_prefix(param_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(param_name.startswith(prefix) for prefix in prefixes)


def is_embedding_parameter(param_name: str) -> bool:
    return param_name in EMBEDDING_PARAM_NAMES or _has_prefix(
        param_name, EMBEDDING_PARAM_PREFIXES) or "word_embeddings" in param_name


def is_lm_head_parameter(param_name: str) -> bool:
    return param_name in LM_HEAD_PARAM_NAMES or _has_prefix(
        param_name, LM_HEAD_PARAM_PREFIXES)


def is_norm_parameter(param_name: str) -> bool:
    return param_name in NORM_PARAM_NAMES or _has_prefix(
        param_name, NORM_PARAM_PREFIXES) or "embeddings.LayerNorm" in param_name


def is_vision_encoder_parameter(param_name: str) -> bool:
    return _has_prefix(param_name, VISION_ENCODER_PARAM_PREFIXES) or any(
        token in param_name for token in (
            "vision", "visual", "image_encoder", "vision_model",
            "vision_tower"))


def is_projector_parameter(param_name: str) -> bool:
    return _has_prefix(param_name, PROJECTOR_PARAM_PREFIXES) or any(
        token in param_name for token in (
            "projector", "connector", "mm_projector",
            "multi_modal_projector"))


def is_pooler_parameter(param_name: str) -> bool:
    return _has_prefix(param_name, POOLER_PARAM_PREFIXES)


def is_shareable_parameter(
    param_name: str,
    config: WeightSharingConfig | None = None,
) -> bool:
    """判断一个参数是否可以跨进程共享。"""
    cfg = _config_or_default(config)

    if any(param_name.endswith(suffix) for suffix in _NON_SHAREABLE_SUFFIXES):
        return False

    if is_embedding_parameter(param_name):
        return cfg.share_embedding
    if is_lm_head_parameter(param_name):
        return cfg.share_lm_head
    if is_norm_parameter(param_name):
        return cfg.share_norm
    if is_vision_encoder_parameter(param_name):
        return cfg.share_vision_encoder
    if is_projector_parameter(param_name):
        return cfg.share_projector
    if is_pooler_parameter(param_name):
        return cfg.share_pooler

    return _has_prefix(param_name, SHAREABLE_PARAM_PREFIXES)


def filter_shareable_param_names(
    param_names: Iterable[str],
    config: WeightSharingConfig | None = None,
) -> set[str]:
    return {
        name for name in param_names if is_shareable_parameter(name, config)
    }


def get_shareable_param_names(
    model,
    config: WeightSharingConfig | None = None,
) -> set[str]:
    """返回 model 中所有可共享参数的名称集合。"""
    return filter_shareable_param_names(
        (name for name, _ in model.named_parameters()), config)


def get_non_shareable_param_names(
    model,
    config: WeightSharingConfig | None = None,
) -> set[str]:
    """返回 model 中所有不可共享参数的名称集合。"""
    return {
        name for name, _ in model.named_parameters()
        if not is_shareable_parameter(name, config)
    }
