# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.platforms import current_platform
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    instanttensor_weights_iterator,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        subfolder: str | None = None
        """The subfolder inside the model repo."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif self.load_config.load_format == "instanttensor":
                weights_iterator = instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get(
                            "num_threads", self.DEFAULT_NUM_THREADS
                        ),
                    )
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering.

        When expert parallelism is active, each rank only needs a subset of
        expert weights.  By computing the set upfront we can skip non-local
        expert tensors *before* reading them from disk.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return
        
        # When EPLB is enabled, redundant physical expert slots may map to
        # logical experts that belong to other ranks in the default partition.
        # The weight loader needs to see ALL logical expert weights so it can
        # populate these redundant slots.  Skip the filter entirely.
        if parallel_config.enable_eplb:
            return

        num_experts = model_config.get_num_experts()
        if num_experts <= 0:
            return

        # EP size/rank computation mirrors FusedMoEParallelConfig.make():
        #   ep_size = dp_size * pcp_size * tp_size (flattened)
        #   ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        if model_config.quantization == "torchao":
            quant_config = get_quant_config(model_config, self.load_config)
            if (
                hasattr(quant_config, "is_checkpoint_torchao_serialized")
                and quant_config.is_checkpoint_torchao_serialized
                and torchao_version_at_least("0.15.0")
            ):
                self.load_config.safetensors_load_strategy = "torchao"

        self._init_ep_weight_filter(model_config)

        weights_to_load = {name for name, _ in model.named_parameters()}

        shareable_names = _get_phase2_shareable_names(model)
        if shareable_names:
            weights_to_load -= shareable_names
            _shrink_shareable_params(model, shareable_names)
            # Patch weight_loader on each shrunken param so that any
            # load attempt (including fused/stacked variants) is a no-op.
            _patch_shareable_weight_loaders(model, shareable_names)
            # Replace fp8 Marlin repack on shareable quantized modules
            # with a no-op that only allocates workspace.  The primary
            # already repacked these weights; the secondary receives the
            # final form via IPC.  This prevents an assertion failure on
            # the 1-element placeholder shape.
            _patch_shareable_quant_postprocess(model, shareable_names)

        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
            scope="local",
        )
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                # Allow skipping shareable params for weight sharing.
                skip_ok = shareable_names or set()
                weights_not_loaded -= skip_ok
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}"
                    )

    @instrument(span_name="Import shared weights from IPC")
    def post_load_weights(self, model: nn.Module,
                          vllm_config: VllmConfig) -> None:
        """Fill shareable parameters from CUDA IPC handles.

        Called after process_weights_after_loading inside
        BaseModelLoader.load_model.  The primary already processed
        (quantised, repacked) the shareable weights; we import them
        directly via IPC.
        """
        ws_config = vllm_config.weight_sharing_config
        if ws_config is None or ws_config.mode != "secondary":
            return
        from vllm.distributed.weight_sharing import WeightSharingManager
        wsm = WeightSharingManager(ws_config)
        imported = wsm.import_weights(model)
        logger.info(
            "WeightSharing: IPC-imported %d shareable params.", len(imported))
        # After import, every shareable quantized layer now has correctly
        # shaped weights on CUDA.  Ensure Marlin workspace is allocated.
        _ensure_workspace(model)


def _get_phase2_shareable_names(model) -> set[str]:
    """Return shareable parameter names for secondary weight-sharing."""
    try:
        from vllm.config import get_current_vllm_config
        ws = get_current_vllm_config().weight_sharing_config
    except Exception:
        return set()
    if ws is None or ws.mode != "secondary":
        return set()
    from vllm.distributed.weight_sharing.parameter_filter import (
        get_shareable_param_names,
    )
    return get_shareable_param_names(model)


def _shrink_shareable_params(model, shareable_names: set[str]) -> None:
    """Move non-shareable params to CUDA and replace shareable with meta.

    Called after CPU-based initialization of a secondary instance.
    Non-shareable params (embed, head, norm) are moved to the CUDA device
    so that load_weights and process_weights_after_loading work normally.
    Shareable params are replaced with meta tensors (zero GPU memory).
    After IPC import they will point directly to primary's GPU memory.
    """
    device = torch.device("cuda")
    params = dict(model.named_parameters())
    n_moved = 0
    n_meta = 0
    for name, param in params.items():
        if name in shareable_names:
            # Keep on CPU as 1-element placeholder — avoids vLLM's
            # BasevLLMParameter.__torch_function__ type check that
            # blocks meta device assignment.  The real IPC tensor
            # will replace this during import_weights().
            param.data = torch.empty(
                1, dtype=param.data.dtype, device=torch.device("cpu"),
            )
            n_meta += 1
        elif param.data.device.type == "cpu":
            param.data = param.data.to(device)
            n_moved += 1
    logger.info(
        "WeightSharing: CPU-init secondary — %d non-shareable params moved "
        "to CUDA, %d shareable params set to meta (0 GPU).", n_moved, n_meta)
    if current_platform.is_cuda():
        logger.info(
            "WeightSharing: GPU memory after init: %d MiB used / %d MiB total",
            (torch.cuda.get_device_properties(0).total_memory
             - torch.cuda.mem_get_info()[0]) // (1024**2),
            torch.cuda.get_device_properties(0).total_memory // (1024**2))


def _patch_shareable_weight_loaders(model, shareable_names: set[str]) -> None:
    """Replace every shareable parameter's weight_loader with a no-op.

    After shrinking, the 1-element placeholder cannot accept the
    full checkpoint tensor.  Since the real weights will be filled
    from IPC handles later, we silence the weight_loader entirely.
    """
    params = dict(model.named_parameters())
    for name in shareable_names:
        if name not in params:
            continue
        param = params[name]
        # Save the original weight_loader for reference (not used now,
        # but allows future undo).
        if not hasattr(param, "_saved_weight_loader"):
            param._saved_weight_loader = getattr(param, "weight_loader", None)
        param.weight_loader = lambda *_, **__: None


def _patch_shareable_quant_postprocess(model, shareable_names: set[str]) -> None:
    """Patch quant_method.process_weights_after_loading for every module
    that owns a shareable parameter (Phase 2 secondary).

    vLLM's quantised linear layers (QKVParallelLinear etc.) do not expose
    a ``.weight`` attribute, so we derive module paths from the parameter
    names provided by the caller.

    The primary already ran ``prepare_fp8_layer_for_marlin`` on these
    layers; the secondary must not re-repack the CPU placeholder.
    Instead, we replace the method with a stub that allocates workspace.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    # Derive module paths from parameter names:
    module_paths: set[str] = set()
    for name in shareable_names:
        module_paths.add(name.rsplit(".", 1)[0])

    for module_path, module in model.named_modules():
        if module_path not in module_paths:
            continue
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None or not hasattr(
            quant_method, "process_weights_after_loading"
        ):
            continue

        setattr(quant_method,
                "_saved_process_weights_after_loading",
                quant_method.process_weights_after_loading)

        def _make_stub(mkw=marlin_make_workspace_new):
            def _stub(layer: torch.nn.Module) -> None:
                if not hasattr(layer, "input_scale"):
                    layer.input_scale = None
                if not hasattr(layer, "workspace"):
                    cuda_dev = torch.device("cuda", torch.cuda.current_device())
                    layer.workspace = mkw(cuda_dev, max_blocks_per_sm=4)
            return _stub

        quant_method.process_weights_after_loading = _make_stub()


def _ensure_workspace(model) -> None:
    """Allocate Marlin workspace for every quantised linear layer
    that does not have one yet.  A safety net in case the
    process_weights_after_loading stub missed a module."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    cuda_dev = torch.device("cuda", torch.cuda.current_device())
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and not hasattr(module, "input_scale"):
            module.input_scale = None
        if quant_method is not None and not hasattr(module, "workspace"):
            module.workspace = marlin_make_workspace_new(cuda_dev, max_blocks_per_sm=4)
