# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import signal
import time
from dataclasses import dataclass

import torch
from torch import nn

import socket

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.distributed.weight_sharing import WeightSharingManager
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.utils.mem_utils import format_gib
from vllm.v1.worker.gpu_worker import (
    init_device_context,
    init_distributed_context,
    requires_distributed_context,
)

logger = init_logger(__name__)


def _ensure_distributed_env() -> None:
    """Ensure MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE / LOCAL_RANK
    are set when using env:// rendezvous for single-GPU primary export."""
    if os.environ.get("MASTER_ADDR", ""):
        return  # already configured (e.g. torchrun or external launcher)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if not os.environ.get("MASTER_PORT", ""):
        with socket.socket() as s:
            s.bind(("", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


@dataclass
class WeightExportResult:
    rank: int
    local_rank: int
    device: str
    model: str
    load_format: str
    dtype: str
    parameter_count: int
    elapsed_seconds: float
    memory_allocated_bytes: int
    memory_reserved_bytes: int


class WeightExportService:
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        rank: int = 0,
        local_rank: int = 0,
        distributed_init_method: str | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.rank = rank
        self.local_rank = local_rank
        self.distributed_init_method = distributed_init_method
        self.model: nn.Module | None = None
        self.manager: WeightSharingManager | None = None
        self.distributed_initialized = False
        self.shutdown_requested = False

    @torch.inference_mode()
    def start(self) -> WeightExportResult:
        ws_config = self.vllm_config.weight_sharing_config
        if ws_config is None or ws_config.mode != "primary":
            raise ValueError(
                "export-weights requires --weight-sharing-config with "
                "mode='primary'."
            )

        start = time.perf_counter()
        device = init_device_context(self.vllm_config, self.local_rank)
        with set_current_vllm_config(self.vllm_config):
            dist_init_required = (
                requires_distributed_context(self.vllm_config)
                or ws_config.mode == "primary"
            )
            if dist_init_required:
                dist_method = self.distributed_init_method or "env://"
                _ensure_distributed_env()
                init_distributed_context(
                    self.vllm_config,
                    self.rank,
                    self.local_rank,
                    dist_method,
                )
                self.distributed_initialized = True

            self.model = get_model(vllm_config=self.vllm_config)
            self.manager = WeightSharingManager(ws_config)
            self.manager.export_weights(self.model)
            torch.accelerator.synchronize(device)
            torch.accelerator.empty_cache()

        result = WeightExportResult(
            rank=self.rank,
            local_rank=self.local_rank,
            device=str(device),
            model=self.vllm_config.model_config.model,
            load_format=str(self.vllm_config.load_config.load_format),
            dtype=str(self.vllm_config.model_config.dtype),
            parameter_count=_parameter_count(self.model),
            elapsed_seconds=time.perf_counter() - start,
            memory_allocated_bytes=_memory_allocated(device),
            memory_reserved_bytes=_memory_reserved(device),
        )
        _log_result(result)
        return result

    def wait_forever(self) -> None:
        logger.info(
            "Weight export service is ready. Keeping primary weights alive."
        )
        while not self.shutdown_requested:
            time.sleep(1)

    def request_shutdown(self, signum: int | None = None) -> None:
        if signum is not None:
            logger.info("Received signal %d, shutting down weight export service.", signum)
        self.shutdown_requested = True

    def shutdown(self) -> None:
        if self.manager is not None:
            self.manager.cleanup()
            self.manager = None
        self.model = None
        gc.collect()
        torch.accelerator.empty_cache()
        if self.distributed_initialized:
            cleanup_dist_env_and_memory()
            self.distributed_initialized = False


def get_external_launcher_rank_info() -> tuple[int, int, int]:
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise ValueError(
            "RANK, LOCAL_RANK, and WORLD_SIZE must be set when "
            f"using --external-launcher; missing {missing}"
        )
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def run_weight_export_service(
    vllm_config: VllmConfig,
    *,
    rank: int = 0,
    local_rank: int = 0,
    distributed_init_method: str | None = None,
) -> WeightExportResult:
    service = WeightExportService(
        vllm_config,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
    )

    def signal_handler(signum, frame):
        service.request_shutdown(signum)

    old_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    old_sigint = signal.signal(signal.SIGINT, signal_handler)
    try:
        result = service.start()
        service.wait_forever()
        return result
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
        service.shutdown()


def _parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _memory_allocated(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.accelerator.memory_allocated(device)
    return 0


def _memory_reserved(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.accelerator.memory_reserved(device)
    return 0


def _log_result(result: WeightExportResult) -> None:
    logger.info(
        "Weight export complete: rank=%d local_rank=%d device=%s "
        "model=%s load_format=%s dtype=%s parameters=%d elapsed=%.2fs "
        "allocated=%s reserved=%s",
        result.rank,
        result.local_rank,
        result.device,
        result.model,
        result.load_format,
        result.dtype,
        result.parameter_count,
        result.elapsed_seconds,
        format_gib(result.memory_allocated_bytes),
        format_gib(result.memory_reserved_bytes),
    )
