# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.vllm.patches import (
    VLLM_ALL_RANGE,
    VLLM_V8_RANGE,
    VLLM_V9_PLUS_RANGE,
    ElasticBlockPoolPatch,
    EngineCorePatch,
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    KVCacheCoordinatorPatch,
    KVCacheManagerPatch,
    PoolingModelRunnerPatch,
)
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_enabled() -> bool:
    ok = os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")
    logger.warning(
        "KVCACHED_AUTOPATCH=%s → autopatch %s",
        os.getenv("KVCACHED_AUTOPATCH", "false"),
        "ENABLED" if ok else "DISABLED",
    )
    return ok


@when_imported("vllm")
def _patch_vllm(_vllm: types.ModuleType) -> None:
    logger.warning(
        "KVCACHED AUTOPATCH v2.1 — patch_initialize_kv_cache REMOVED, "
        "using lazy-init in execute_model. vLLM version=%s",
        getattr(_vllm, "__version__", "unknown"),
    )

    if not _env_enabled():
        logger.warning("KVCACHED_AUTOPATCH not set, skipping all patches")
        return

    # Create patch manager and register version-specific vLLM patches
    patch_manager = PatchManager("vllm")

    patches = [
        (ElasticBlockPoolPatch(), VLLM_ALL_RANGE),
        (EngineCorePatch(), VLLM_ALL_RANGE),
        (GPUModelRunnerPatch(), VLLM_ALL_RANGE),
        (GPUWorkerPatch(), VLLM_ALL_RANGE),
        (KVCacheCoordinatorPatch(), VLLM_V9_PLUS_RANGE),
        (KVCacheManagerPatch(), VLLM_V8_RANGE),
        (PoolingModelRunnerPatch(), VLLM_ALL_RANGE),
    ]
    logger.warning("Registering %d patches for vLLM", len(patches))
    patch_manager.register_patches_with_versions(patches)

    # Apply all patches
    logger.warning("Applying all patches...")
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("vllm", results)
    logger.warning("Autopatch complete: %d succeeded, %d failed",
                   sum(1 for v in results.values() if v),
                   sum(1 for v in results.values() if not v))
