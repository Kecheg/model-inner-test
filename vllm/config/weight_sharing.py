# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightSharingConfig:
    """跨进程模型权重共享配置。

    启用后，同一节点上运行相同模型的多个 vLLM 进程可以通过 CUDA IPC handle
    共享解码器层权重，显著降低显存占用。
    嵌入层、lm_head 和 KV cache 不参与共享。
    """

    mode: Literal["off", "primary", "secondary"] = "off"
    """权重共享模式，三选一：

    - ``off``：不启用（默认），进程独立运行，无任何共享。
    - ``primary``：主进程，完整加载模型后将共享参数的 CUDA IPC handle
      写入注册表文件，必须先于 secondary 启动。
    - ``secondary``：从进程，使用 CPU 初始化模型（0 GPU 峰值），
      然后从注册表读取 IPC handle，直接指向 primary 的 GPU 内存，
      不从磁盘加载共享参数，节省大量 GPU 显存。
    """

    registry_path: str = "/dev/shm/vllm_weight_share"
    """IPC handle 注册表目录路径。
    默认使用 tmpfs（/dev/shm），速度快且重启后自动清理。
    多个 Pod 共享时必须通过 hostPath 挂载相同的宿主机目录。
    实际路径格式：{registry_path}/{model_id}/pp{pp_rank}_tp{tp_rank}/
    """

    timeout_seconds: int = 300
    """secondary 等待 primary 写入注册表的最大时间（秒）。
    超时后 secondary 启动失败并报错。
    """
