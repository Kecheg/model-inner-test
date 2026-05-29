# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Tests for activation memory pooling (ActivationFTensor + ActivationPoolManager)."""

import pytest
import torch

from kvcached.utils import PAGE_SIZE


class TestActivationPoolManager:
    """Per-model activation memory lifecycle tests."""

    @pytest.fixture(autouse=True)
    def _ensure_kvcached_initialized(self):
        from kvcached import vmm_ops

        vmm_ops.init_kvcached("cuda:0", PAGE_SIZE, False)
        yield
        vmm_ops.shutdown_kvcached()

    def test_initialize_creates_and_stores_tensor(self):
        """initialize() creates a VA tensor and stores the reference."""
        from kvcached.activation_pool_manager import ActivationPoolManager

        peak_bytes = 64 * 1024 * 1024
        pool = ActivationPoolManager(
            model_name="test_embedding",
            peak_activation_bytes=peak_bytes,
            dtype=torch.float16,
            device="cuda:0",
        )
        pool.initialize()

        assert pool.is_initialized
        assert not pool.is_mapped
        assert pool.tensor is not None
        assert pool.tensor.data_ptr() != 0
        assert pool.tensor.numel() * pool.tensor.element_size() == peak_bytes

    def test_begin_end_forward_lifecycle(self):
        """begin_forward() maps; end_forward() unmaps."""
        from kvcached.activation_pool_manager import ActivationPoolManager

        peak_bytes = 64 * 1024 * 1024
        pool = ActivationPoolManager(
            model_name="test_lifecycle",
            peak_activation_bytes=peak_bytes,
            dtype=torch.float16,
            device="cuda:0",
        )
        pool.initialize()

        pool.begin_forward()
        assert pool.is_mapped

        pool.end_forward()
        assert not pool.is_mapped

    def test_begin_forward_idempotent(self):
        """Double begin_forward is safe."""
        from kvcached.activation_pool_manager import ActivationPoolManager

        peak_bytes = 64 * 1024 * 1024
        pool = ActivationPoolManager(
            model_name="test_idem",
            peak_activation_bytes=peak_bytes,
            dtype=torch.float16,
            device="cuda:0",
        )
        pool.initialize()

        pool.begin_forward()
        pool.begin_forward()
        assert pool.is_mapped

        pool.end_forward()

    def test_end_forward_before_begin(self):
        """end_forward before begin_forward is a safe no-op."""
        from kvcached.activation_pool_manager import ActivationPoolManager

        peak_bytes = 64 * 1024 * 1024
        pool = ActivationPoolManager(
            model_name="test_noop",
            peak_activation_bytes=peak_bytes,
            dtype=torch.float16,
            device="cuda:0",
        )
        pool.initialize()
        pool.end_forward()
        assert not pool.is_mapped

    def test_tensor_accessible_while_mapped(self):
        """VMM tensor is readable/writable after begin_forward."""
        from kvcached.activation_pool_manager import ActivationPoolManager

        peak_bytes = 64 * 1024 * 1024
        pool = ActivationPoolManager(
            model_name="test_access",
            peak_activation_bytes=peak_bytes,
            dtype=torch.float16,
            device="cuda:0",
        )
        pool.initialize()

        pool.begin_forward()
        # Tensor is physically backed; reading/writing should not fault.
        _ = pool.tensor[:1024].sum().item()
        pool.tensor[0] = 1.0
        pool.end_forward()

    def test_repeated_forward_cycles(self):
        """Multiple map/unmap cycles succeed."""
        from kvcached.activation_pool_manager import ActivationPoolManager

        peak_bytes = 64 * 1024 * 1024
        pool = ActivationPoolManager(
            model_name="test_repeat",
            peak_activation_bytes=peak_bytes,
            dtype=torch.float16,
            device="cuda:0",
        )
        pool.initialize()

        for _ in range(10):
            pool.begin_forward()
            assert pool.is_mapped
            pool.end_forward()
            assert not pool.is_mapped


class TestActivationCoordinator:
    """Multi-model coordination tests."""

    @pytest.fixture(autouse=True)
    def _ensure_kvcached_initialized(self):
        from kvcached import vmm_ops

        vmm_ops.init_kvcached("cuda:0", PAGE_SIZE, False)
        yield
        vmm_ops.shutdown_kvcached()

    def test_register_multiple_models(self):
        from kvcached.activation_pool_manager import ActivationCoordinator

        coordinator = ActivationCoordinator()
        models = ["embedding_v1", "rerank_v2", "multilingual"]

        for name in models:
            pool = coordinator.register_model(
                name, 64 * 1024 * 1024, torch.float16, "cuda:0"
            )
            assert pool.is_initialized
            assert not pool.is_mapped

        assert set(coordinator.model_names) == set(models)
        assert coordinator.total_peak_bytes == len(models) * 64 * 1024 * 1024

    def test_independent_forward_cycles(self):
        from kvcached.activation_pool_manager import ActivationCoordinator

        coordinator = ActivationCoordinator()
        coordinator.register_model("a", 32 * 1024 * 1024, torch.float16, "cuda:0")
        coordinator.register_model("b", 48 * 1024 * 1024, torch.float16, "cuda:0")

        coordinator.begin_forward("a")
        assert coordinator.get_pool("a").is_mapped
        assert not coordinator.get_pool("b").is_mapped
        coordinator.end_forward("a")

        coordinator.begin_forward("b")
        assert not coordinator.get_pool("a").is_mapped
        assert coordinator.get_pool("b").is_mapped
        coordinator.end_forward("b")

    def test_register_existing_returns_same_pool(self):
        from kvcached.activation_pool_manager import ActivationCoordinator

        coordinator = ActivationCoordinator()
        p1 = coordinator.register_model("x", 32 * 1024 * 1024, torch.float16, "cuda:0")
        p2 = coordinator.register_model("x", 64 * 1024 * 1024, torch.float16, "cuda:0")
        assert p1 is p2

    def test_release_all(self):
        from kvcached.activation_pool_manager import ActivationCoordinator

        coordinator = ActivationCoordinator()
        for name in ["m1", "m2", "m3"]:
            coordinator.register_model(name, 32 * 1024 * 1024, torch.float16, "cuda:0")
            coordinator.begin_forward(name)

        coordinator.release_all()
        for name in ["m1", "m2", "m3"]:
            assert not coordinator.get_pool(name).is_mapped

    def test_separate_vmm_ranges(self):
        from kvcached.activation_pool_manager import ActivationCoordinator

        coordinator = ActivationCoordinator()
        coordinator.register_model("x", 32 * 1024 * 1024, torch.float16, "cuda:0")
        coordinator.register_model("y", 64 * 1024 * 1024, torch.float16, "cuda:0")

        px = coordinator.get_pool("x")
        py = coordinator.get_pool("y")
        assert px.tensor.data_ptr() != py.tensor.data_ptr()
        assert px.peak_activation_bytes == 32 * 1024 * 1024
        assert py.peak_activation_bytes == 64 * 1024 * 1024


class TestActivationPageAlignment:
    """Page-alignment tests."""

    @pytest.fixture(autouse=True)
    def _ensure_kvcached_initialized(self):
        from kvcached import vmm_ops

        vmm_ops.init_kvcached("cuda:0", PAGE_SIZE, False)
        yield
        vmm_ops.shutdown_kvcached()

    def test_unaligned_rounded_up(self):
        from kvcached.activation_pool_manager import ActivationCoordinator

        unaligned = PAGE_SIZE + 1024
        coordinator = ActivationCoordinator()
        pool = coordinator.register_model("align", unaligned, torch.float16, "cuda:0")

        expected = (unaligned + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
        assert pool.peak_activation_bytes == expected
        assert pool.peak_activation_bytes % PAGE_SIZE == 0
