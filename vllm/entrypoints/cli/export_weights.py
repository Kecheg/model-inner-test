# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import socket
import subprocess
import sys

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.weight_exporter import (
    get_external_launcher_rank_info,
    run_weight_export_service,
)

logger = init_logger(__name__)

DESCRIPTION = """Load model weights, export CUDA IPC handles, and stay alive.

This command starts a primary weight-sharing process without starting the vLLM
serving engine, scheduler, KV cache, warmup, CUDA graph capture, or HTTP server.
Secondary vLLM services can reuse these weights with
--weight-sharing-config '{"mode":"secondary"}'.
"""


class ExportWeightsSubcommand(CLISubcommand):
    """The `export-weights` subcommand for vLLM CLI."""

    name = "export-weights"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        engine_args = EngineArgs.from_cli_args(args)
        engine_args.enforce_eager = True
        engine_args.disable_log_stats = True
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.ENGINE_CONTEXT
        )

        ws_config = vllm_config.weight_sharing_config
        if ws_config is None or ws_config.mode != "primary":
            raise ValueError(
                "export-weights requires --weight-sharing-config "
                "'{\"mode\":\"primary\"}'."
            )

        if args.external_launcher:
            rank, local_rank, world_size = get_external_launcher_rank_info()
            expected_world_size = _expected_world_size(vllm_config)
            if world_size != expected_world_size:
                raise ValueError(
                    f"WORLD_SIZE ({world_size}) must match configured parallel "
                    f"world size ({expected_world_size})."
                )
            run_weight_export_service(
                vllm_config,
                rank=rank,
                local_rank=local_rank,
                distributed_init_method="env://",
            )
            return

        world_size = _expected_world_size(vllm_config)
        if world_size == 1:
            run_weight_export_service(vllm_config)
            return

        # Auto-launch: spawn child processes for distributed export.
        # This replaces the need for torchrun when PP>1 or TP>1.
        _launch_distributed_export(vllm_config, world_size)

    def validate(self, args: argparse.Namespace) -> None:
        # Removed the torchrun requirement — cmd() now auto-launches
        # child processes for distributed configs internally.
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Export model weights for CUDA IPC sharing.",
            description=DESCRIPTION,
            usage="vllm export-weights [model_tag] [options]",
        )
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "model_tag",
            nargs="?",
            default=None,
            help="Model tag or path. If provided, it overrides --model.",
        )
        parser.add_argument(
            "--external-launcher",
            action="store_true",
            help="Use rank information from torchrun or another external launcher.",
        )
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def _expected_world_size(vllm_config) -> int:
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
        * parallel_config.decode_context_parallel_size
    )


def _launch_distributed_export(vllm_config, world_size: int) -> None:
    """Launch weight export child processes, replacing the need for torchrun.

    The current process becomes rank 0. Child processes are spawned for
    ranks 1..N-1, each re-invoking this CLI with --external-launcher.
    """
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "")
    if not master_port:
        with socket.socket() as s:
            s.bind(("", 0))
            master_port = str(s.getsockname()[1])

    logger.info(
        "Auto-launching %d worker(s) for distributed weight export "
        "(master=%s:%s).",
        world_size - 1, master_addr, master_port,
    )

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
        run_weight_export_service(
            vllm_config,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
        )
    finally:
        for p in children:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def cmd_init() -> list[CLISubcommand]:
    return [ExportWeightsSubcommand()]
