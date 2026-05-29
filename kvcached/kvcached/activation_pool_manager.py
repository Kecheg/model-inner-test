# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Activation memory pool manager for Embedding/Rerank models.

Pooling models (embedding / rerank) allocate large temporary activation tensors
during each forward pass but have no generative KV cache.  We pre-allocate a
CUDA-VMM-backed virtual address region sized to the model's peak activation
footprint and map / unmap physical pages around each forward.  This returns
physical GPU memory to the shared VMM pool after the forward so the KV cache
(or another model) can claim it.

The model itself continues to use PyTorch's native allocator.  However PyTorch
with ``expandable_segments=True`` also allocates physical pages through the
same CUDA VMM pool (``cuMemCreate``).  The VMM map/unmap cycle therefore acts
as a physical-page *budget* that forces the pool to release pages after every
forward.

Under high concurrency the per-request ``cuMemCreate`` / ``cuMemRelease`` cost
is significant.  Use ``KVCACHED_ACTIVATION_KEEP_MAPPED_MS`` to keep pages
mapped between back-to-back requests within a time window.

Environment variables:
  KVCACHED_ACTIVATION_POOL_ENABLED   Enable activation memory pooling (default: "true")
  KVCACHED_ACTIVATION_KEEP_MAPPED_MS Keep pages mapped after forward (ms, default: 0)
"""

import os
import threading
from typing import Dict, Optional

import torch

from kvcached.utils import PAGE_SIZE, align_to, get_kvcached_logger

logger = get_kvcached_logger(__name__)

_ACTIVATION_POOL_ENABLED = os.getenv(
    "KVCACHED_ACTIVATION_POOL_ENABLED", "true"
).lower() in ("true", "1")

_KEEP_MAPPED_MS = int(os.getenv("KVCACHED_ACTIVATION_KEEP_MAPPED_MS", "0"))


def is_activation_pool_enabled() -> bool:
    return _ACTIVATION_POOL_ENABLED


class ActivationPoolManager:
    """Manages activation memory for a single Embedding/Rerank model.

    Uses CUDA VMM to virtualize activation memory:

    - ``initialize()``: reserves virtual address space (zero-page-backed).
    - ``begin_forward()``: maps physical pages into the reserved VA region.
    - ``end_forward()``: if *keep_mapped_ms* is set, leaves pages mapped so
      the next forward can reuse them without re-allocation.  Otherwise
      unmaps immediately.
    """

    _keep_mapped_ms: int = _KEEP_MAPPED_MS

    def __init__(
        self,
        model_name: str,
        peak_activation_bytes: int,
        dtype: torch.dtype,
        device: str,
        group_id: int = 0,
    ):
        self.model_name = model_name
        self.peak_activation_bytes = peak_activation_bytes
        self.dtype = dtype
        self.device = device
        self.group_id = group_id
        self._tensor_name = f"activation_{model_name}"
        self._initialized = False
        self._mapped = False
        self._tensor: Optional[torch.Tensor] = None
        self._unmap_timer: Optional[threading.Timer] = None

    @classmethod
    def set_keep_mapped_ms(cls, ms: int) -> None:
        cls._keep_mapped_ms = ms
        logger.warning(
            "ActivationPoolManager keep_mapped_ms set to %d ms", ms,
        )

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        if self._initialized:
            return

        from kvcached import vmm_ops

        logger.warning(
            "[%s] Activation pool init: reserving VA space (%d MB, dtype=%s, "
            "device=%s, keep_mapped_ms=%d)",
            self.model_name,
            self.peak_activation_bytes // (1024 * 1024),
            self.dtype,
            self.device,
            ActivationPoolManager._keep_mapped_ms,
        )
        self._tensor = vmm_ops.create_activation_tensor(
            self.peak_activation_bytes,
            self.dtype,
            self.device,
            self._tensor_name,
            self.group_id,
        )
        self._initialized = True
        logger.warning(
            "[%s] VA reserved (ptr=0x%x, nbytes=%d)",
            self.model_name,
            self._tensor.data_ptr(),
            self._tensor.numel() * self._tensor.element_size(),
        )

    def begin_forward(self) -> None:
        if not self._initialized:
            return
        if self._mapped:
            # Pages already mapped (timer hasn't fired yet) — cancel the
            # pending unmap and keep going.
            self._cancel_unmap_timer()
            logger.debug(
                "[%s] begin_forward: pages already mapped, cancelling "
                "pending unmap timer",
                self.model_name,
            )
            return

        from kvcached import vmm_ops

        self._cancel_unmap_timer()
        torch.cuda.empty_cache()

        num_pages = self.peak_activation_bytes // PAGE_SIZE
        logger.warning(
            "[%s] begin_forward: mapping %d pages (%d MB), device=%s",
            self.model_name,
            num_pages,
            self.peak_activation_bytes // (1024 * 1024),
            self.device,
        )
        vmm_ops.map_activation(self._tensor_name, self.group_id)
        self._mapped = True
        logger.warning(
            "[%s] begin_forward: physical pages mapped (ptr=0x%x, nbytes=%d)",
            self.model_name,
            self._tensor.data_ptr(),
            self._tensor.numel() * self._tensor.element_size(),
        )

    def end_forward(self, force: bool = False) -> None:
        """Release physical pages after a forward pass.

        If *keep_mapped_ms* > 0 and *force* is False, starts a background
        timer that will unmap after the window expires.  An incoming
        ``begin_forward`` before the timer fires cancels it, avoiding the
        unmap/re-map overhead.
        """
        if not self._initialized or not self._mapped:
            return

        if not force and ActivationPoolManager._keep_mapped_ms > 0:
            self._schedule_unmap_timer()
            logger.debug(
                "[%s] end_forward: keeping pages mapped, scheduled unmap "
                "in %d ms",
                self.model_name,
                ActivationPoolManager._keep_mapped_ms,
            )
            return

        self._unmap()

    # -- timer helpers ----------------------------------------------------

    def _schedule_unmap_timer(self) -> None:
        self._cancel_unmap_timer()
        delay_s = ActivationPoolManager._keep_mapped_ms / 1000.0
        self._unmap_timer = threading.Timer(delay_s, self._on_unmap_timer)
        self._unmap_timer.daemon = True
        self._unmap_timer.start()

    def _cancel_unmap_timer(self) -> None:
        if self._unmap_timer is not None:
            self._unmap_timer.cancel()
            self._unmap_timer = None

    def _on_unmap_timer(self) -> None:
        logger.warning(
            "[%s] keep_mapped_ms timer fired, auto-releasing pages",
            self.model_name,
        )
        self._unmap()

    # -- low-level unmap --------------------------------------------------

    def _unmap(self) -> None:
        if not self._mapped:
            return

        self._cancel_unmap_timer()
        # Release PyTorch cached memory before unmapping VMM pages so
        # PyTorch doesn't hold references to physical pages we're about
        # to release back to the GPU pool.
        torch.cuda.empty_cache()

        from kvcached import vmm_ops

        num_pages = self.peak_activation_bytes // PAGE_SIZE
        logger.warning(
            "[%s] end_forward: unmapping %d pages (%d MB) back to pool",
            self.model_name,
            num_pages,
            self.peak_activation_bytes // (1024 * 1024),
        )
        vmm_ops.unmap_activation(self._tensor_name, self.group_id)
        self._mapped = False
        logger.warning(
            "[%s] end_forward: physical pages released, memory returned to pool",
            self.model_name,
        )

    # -- properties --------------------------------------------------------

    @property
    def is_mapped(self) -> bool:
        return self._mapped

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return self._tensor


class ActivationCoordinator:
    """Coordinates activation memory across multiple pooling models.

    Manages multiple :class:`ActivationPoolManager` instances for co-deployed
    Embedding / Rerank models.
    """

    def __init__(self):
        self._pools: Dict[str, ActivationPoolManager] = {}

    def register_model(
        self,
        model_name: str,
        peak_activation_bytes: int,
        dtype: torch.dtype,
        device: str,
        group_id: int = 0,
    ) -> ActivationPoolManager:
        if model_name in self._pools:
            return self._pools[model_name]

        aligned_bytes = align_to(peak_activation_bytes, PAGE_SIZE)
        if aligned_bytes != peak_activation_bytes:
            logger.warning(
                "[coordinator] Aligning '%s': %d -> %d (%d pages)",
                model_name,
                peak_activation_bytes,
                aligned_bytes,
                aligned_bytes // PAGE_SIZE,
            )

        pool = ActivationPoolManager(
            model_name=model_name,
            peak_activation_bytes=aligned_bytes,
            dtype=dtype,
            device=device,
            group_id=group_id,
        )
        pool.initialize()
        self._pools[model_name] = pool
        logger.warning(
            "[coordinator] Registered '%s': peak=%d MB, total models=%d, "
            "total peak=%d MB",
            model_name,
            aligned_bytes // (1024 * 1024),
            len(self._pools),
            self.total_peak_bytes // (1024 * 1024),
        )
        return pool

    def get_pool(self, model_name: str) -> Optional[ActivationPoolManager]:
        return self._pools.get(model_name)

    def begin_forward(self, model_name: str) -> None:
        pool = self._pools.get(model_name)
        if pool is None:
            logger.warning("[coordinator] Unknown model '%s'", model_name)
            return
        pool.begin_forward()

    def end_forward(self, model_name: str) -> None:
        pool = self._pools.get(model_name)
        if pool is None:
            return
        pool.end_forward()

    def release_all(self) -> None:
        """Force-unmap all models (ignores keep_mapped_ms)."""
        logger.warning(
            "[coordinator] Force-releasing all (%d models, %d MB total)",
            len(self._pools),
            self.total_peak_bytes // (1024 * 1024),
        )
        for name, pool in self._pools.items():
            if pool.is_mapped:
                pool.end_forward(force=True)

    @property
    def total_peak_bytes(self) -> int:
        return sum(p.peak_activation_bytes for p in self._pools.values())

    @property
    def model_names(self):
        return list(self._pools.keys())
