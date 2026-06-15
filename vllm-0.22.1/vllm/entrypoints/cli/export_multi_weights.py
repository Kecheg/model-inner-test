# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import socket
import subprocess
import sys
from copy import deepcopy
from typing import Any

from vllm.config.weight_sharing import WeightSharingConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.weight_exporter import (
    get_external_launcher_rank_info,
    run_multi_weight_export_service,
)

logger = init_logger(__name__)

DESCRIPTION = """Load and export multiple models from one primary process group.

The command keeps one TP/PP primary rank process per rank.  Each rank loads all
configured models' local shard and exports each model's CUDA IPC handles into the
same registry root, separated by the existing per-model registry directory.
"""


class ExportMultiWeightsSubcommand(CLISubcommand):
    """The `export-multi-weights` subcommand for vLLM CLI."""

    name = "export-multi-weights"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_specs = _load_models_json(args.models_json)
        if not model_specs:
            raise ValueError("--models-json must contain at least one model spec.")

        vllm_configs = [
            _create_model_config(args, model_spec) for model_spec in model_specs
        ]

        if args.external_launcher:
            rank, local_rank, world_size = get_external_launcher_rank_info()
            expected_world_size = _expected_world_size(vllm_configs[0])
            if world_size != expected_world_size:
                raise ValueError(
                    f"WORLD_SIZE ({world_size}) must match configured parallel "
                    f"world size ({expected_world_size}).")
            run_multi_weight_export_service(
                vllm_configs,
                rank=rank,
                local_rank=local_rank,
                distributed_init_method="env://",
            )
            return

        world_size = _expected_world_size(vllm_configs[0])
        if world_size == 1:
            run_multi_weight_export_service(vllm_configs)
            return

        _launch_distributed_export(args, world_size)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Export multiple models for CUDA IPC sharing.",
            description=DESCRIPTION,
            usage="vllm export-multi-weights --models-json FILE [options]",
        )
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--models-json",
            required=True,
            help=(
                "JSON file containing a list of model specs.  Each spec must "
                "include 'model'.  Optional spec keys override common CLI args: "
                "served_model_name, tokenizer, tokenizer_mode, trust_remote_code, "
                "dtype, load_format, and share_* flags."),
        )
        parser.add_argument(
            "--registry-path",
            required=True,
            help="Shared registry root for all exported model IPC handles.",
        )
        parser.add_argument(
            "--external-launcher",
            action="store_true",
            help="Use rank information from torchrun or another external launcher.",
        )
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def _load_models_json(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--models-json must contain a JSON list.")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"model spec #{index} must be a JSON object.")
        if "model" not in item:
            raise ValueError(f"model spec #{index} is missing required key 'model'.")
    return data


def _create_model_config(
    args: argparse.Namespace,
    model_spec: dict[str, Any],
):
    model_args = deepcopy(args)
    model_args.model = model_spec["model"]
    model_args.model_tag = None

    # The CLI-only fields for this subcommand must not be copied into
    # EngineArgs.  Deleting them avoids accidentally passing unknown state if
    # EngineArgs grows similarly named fields later.
    for cli_only_key in ("models_json", "registry_path", "external_launcher"):
        if hasattr(model_args, cli_only_key):
            delattr(model_args, cli_only_key)

    for key in (
        "served_model_name",
        "tokenizer",
        "tokenizer_mode",
        "trust_remote_code",
        "dtype",
        "load_format",
        "revision",
        "tokenizer_revision",
        "code_revision",
        "max_model_len",
    ):
        if key in model_spec:
            setattr(model_args, key, model_spec[key])

    model_args.weight_sharing_config = _build_weight_sharing_config(
        args.registry_path, model_spec)
    engine_args = EngineArgs.from_cli_args(model_args)
    engine_args.enforce_eager = True
    engine_args.disable_log_stats = True
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.ENGINE_CONTEXT)
    return vllm_config


def _build_weight_sharing_config(
    registry_path: str,
    model_spec: dict[str, Any],
) -> WeightSharingConfig:
    config_data: dict[str, Any] = {
        "mode": "primary",
        "registry_path": registry_path,
    }
    nested = model_spec.get("weight_sharing_config")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("weight_sharing_config must be a JSON object.")
        config_data.update(nested)

    for key in (
        "timeout_seconds",
        "share_embedding",
        "share_lm_head",
        "share_norm",
        "share_vision_encoder",
        "share_projector",
        "share_pooler",
    ):
        if key in model_spec:
            config_data[key] = model_spec[key]

    config_data["mode"] = "primary"
    config_data["registry_path"] = registry_path
    return WeightSharingConfig(**config_data)


def _expected_world_size(vllm_config) -> int:
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
        * parallel_config.decode_context_parallel_size
    )


def _launch_distributed_export(args: argparse.Namespace, world_size: int) -> None:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "")
    if not master_port:
        with socket.socket() as s:
            s.bind(("", 0))
            master_port = str(s.getsockname()[1])

    logger.info(
        "Auto-launching %d worker(s) for multi-model weight export "
        "(master=%s:%s).",
        world_size - 1, master_addr, master_port)

    cmd = [sys.executable] + sys.argv + ["--external-launcher"]
    children: list[subprocess.Popen] = []
    for rank in range(1, world_size):
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world_size)
        env["MASTER_ADDR"] = master_addr
        env["MASTER_PORT"] = master_port
        children.append(subprocess.Popen(cmd, env=env))

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    try:
        ExportMultiWeightsSubcommand.cmd(argparse.Namespace(**{
            **vars(args),
            "external_launcher": True,
        }))
    finally:
        for p in children:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def cmd_init() -> list[CLISubcommand]:
    return [ExportMultiWeightsSubcommand()]
