# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Activation memory pool manager for Embedding/Rerank models.

Provides CUDA VMM-based virtualized activation memory management.  Each model
gets a reserved VA region that is physically backed only during forward passes.
"""

import logging
from typing import Dict, Optional

import torch

from kvcached.utils import PAGE_SIZE, align_to, get_kvcached_logger

logger = get_kvcached_logger(__name__)


class ActivationPoolManager:
    """Manages activation memory for a single Embedding/Rerank model.

    Uses CUDA VMM to virtualize activation memory:
    - On initialize(): reserves virtual address space (no physical pages)
    - On begin_forward(): maps physical pages
    - On end_forward(): unmaps physical pages, returning memory to GPU

    Multiple ActivationPoolManager instances can coexist, each managing
    activation memory for a different model. They share the same underlying
    GPU physical memory pool through CUDA VMM.
    """

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

    def initialize(self) -> None:
        """Reserve virtual address space for activation memory.

        Does NOT allocate physical pages. The tensor view is backed by zero
        pages and must not be written to until begin_forward() is called.
        """
        if self._initialized:
            return

        from kvcached import vmm_ops

        vmm_ops.create_activation_tensor(
            self.peak_activation_bytes,
            self.dtype,
            self.device,
            self._tensor_name,
            self.group_id,
        )
        self._initialized = True
        logger.info(
            "ActivationPoolManager[%s]: VA reserved (%d MB)",
            self.model_name,
            self.peak_activation_bytes // (1024 * 1024),
        )

    def begin_forward(self) -> None:
        """Map physical pages before a forward pass.

        Must be called before the model's forward pass. After this returns,
        the activation memory is physically backed and ready for use.
        """
        if not self._initialized:
            logger.warning(
                "ActivationPoolManager[%s]: begin_forward called before initialize",
                self.model_name,
            )
            return
        if self._mapped:
            return

        from kvcached import vmm_ops

        vmm_ops.map_activation(self._tensor_name, self.group_id)
        self._mapped = True

    def end_forward(self) -> None:
        """Unmap physical pages after a forward pass.

        Must be called after the model's forward pass completes.  This releases
        physical GPU memory back to the pool so other models (or KV Cache) can
        use it.
        """
        if not self._initialized:
            return
        if not self._mapped:
            return

        from kvcached import vmm_ops

        vmm_ops.unmap_activation(self._tensor_name, self.group_id)
        self._mapped = False

    @property
    def is_mapped(self) -> bool:
        return self._mapped

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class ActivationCoordinator:
    """Coordinates activation memory across multiple pooling models.

    Manages multiple ActivationPoolManager instances for co-deployed
    Embedding/Rerank models. Provides:

    - Per-model activation pool lifecycle (init / begin_forward / end_forward)
    - Physical memory budget tracking across all models
    - Batching optimization: keeps pages mapped when back-to-back requests
      for the same model are expected (controlled by keep_mapped_ms)
    """

    def __init__(
        self,
        keep_mapped_ms: float = 0.0,
    ):
        self._pools: Dict[str, ActivationPoolManager] = {}
        self._keep_mapped_ms = keep_mapped_ms

    def register_model(
        self,
        model_name: str,
        peak_activation_bytes: int,
        dtype: torch.dtype,
        device: str,
        group_id: int = 0,
    ) -> ActivationPoolManager:
        """Register a pooling model and create its activation pool.

        Args:
            model_name: Unique name for the model.
            peak_activation_bytes: Peak activation memory size (from profiling).
            dtype: Model computation dtype.
            device: CUDA device string (e.g. "cuda:0").
            group_id: VMM allocator group id.

        Returns:
            The ActivationPoolManager for the registered model.
        """
        if model_name in self._pools:
            return self._pools[model_name]

        # Align to page size
        aligned_bytes = align_to(peak_activation_bytes, PAGE_SIZE)
        if aligned_bytes != peak_activation_bytes:
            logger.info(
                "ActivationCoordinator[%s]: aligned peak bytes %d -> %d",
                model_name,
                peak_activation_bytes,
                aligned_bytes,
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
        return pool

    def get_pool(self, model_name: str) -> Optional[ActivationPoolManager]:
        """Get the ActivationPoolManager for a registered model."""
        return self._pools.get(model_name)

    def begin_forward(self, model_name: str) -> None:
        """Map physical pages for a model's forward pass."""
        pool = self._pools.get(model_name)
        if pool is None:
            logger.warning(
                "ActivationCoordinator: unknown model '%s'", model_name
            )
            return
        pool.begin_forward()

    def end_forward(self, model_name: str) -> None:
        """Unmap physical pages after a model's forward pass."""
        pool = self._pools.get(model_name)
        if pool is None:
            return
        pool.end_forward()

    def release_all(self) -> None:
        """Unmap all activation memory across all models.

        Useful when putting all pooling models to sleep.
        """
        for pool in self._pools.values():
            pool.end_forward()

    @property
    def total_peak_bytes(self) -> int:
        """Sum of peak activation bytes across all registered models."""
        return sum(p.peak_activation_bytes for p in self._pools.values())

    @property
    def model_names(self):
        return list(self._pools.keys())
