# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""基于文件系统的 CUDA IPC handle 注册表。

核心设计：
  Primary 进程将 IPC handle 写入 /dev/shm 下的目录。
  Secondary 进程读取注册表，从 IPC handle 重建 tensor，指向 primary 的 GPU 内存。

进程存活检测（OS flock 方案）：
  利用 Linux 内核保证——进程死亡时（含 SIGKILL）持有的所有 flock 自动释放。
  Primary 启动时获取 primary.lock 的排他锁（LOCK_EX），保持 fd 打开直到进程退出。
  Secondary 检查 primary 存活时尝试非阻塞获取同一锁，若拿到锁→primary 已死。
  这不需要 atexit 或 signal handler，覆盖所有退出方式。
"""

import base64
import fcntl          # 操作系统级文件锁（flock）
import json
import os
import os.path
import pickle
import time
from io import IOBase
from pathlib import Path
from typing import Any, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# 注册表目录下的三个文件
REGISTRY_FILENAME = "weight_registry.json"   # IPC handle + 模型元数据
READY_FILENAME = "primary_ready.signal"       # primary 就绪标记（空文件）
LOCK_FILENAME = "primary.lock"                # flock 进程存活锁

# reduce_tensor 返回的类型：(函数, 参数元组)
IpcHandle = tuple[Any, tuple[Any, ...]]


class WeightSharingRegistry:
    """文件系统注册表管理器。

    注册表目录结构：
      registry_path/
        ├── weight_registry.json   ← Primary 写入的 IPC handle 和元数据
        ├── primary_ready.signal   ← Primary 写入注册表后创建
        └── primary.lock           ← Primary 获取排他 flock，进程生命周期

    存活约定：
      Primary 在写入注册表之前获取 primary.lock 的 LOCK_EX。
      Secondary 尝试对同一文件做 LOCK_EX | LOCK_NB：
        - 被阻塞（EWOULDBLOCK）→ primary 持有锁 → primary 存活 ✅
        - 成功获取                     → primary 已死 → 注册表内容可能过期 ❌
    """

    def __init__(self, registry_path: str) -> None:
        self._base_path = Path(registry_path)
        self._registry_file = self._base_path / REGISTRY_FILENAME
        self._ready_file = self._base_path / READY_FILENAME
        self._lock_file = self._base_path / LOCK_FILENAME
        # primary 持有锁的文件描述符，进程存活期间保持打开
        self._lock_fd: Optional[IOBase] = None

    # ------------------------------------------------------------------
    # 目录/文件管理
    # ------------------------------------------------------------------

    def ensure_directory(self) -> None:
        """创建注册表目录，权限 777（跨用户/容器可访问）。"""
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._base_path.chmod(0o777)

    def write_registry(self, data: dict[str, Any]) -> None:
        """原子写入注册表：先写临时文件，flush+fsync 后 rename。

        两次 fsync：一次文件数据，一次目录元数据。
        保证系统崩溃时不会留下截断的 JSON 文件。
        """
        self.ensure_directory()
        tmp = self._registry_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
            f.flush()                    # 刷新 C 库缓冲区
            os.fsync(f.fileno())         # 强制写入磁盘（内核页缓存落盘）
        os.rename(tmp, self._registry_file)  # 原子替换
        dir_fd = os.open(self._base_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)             # 确保目录元数据也落盘
        finally:
            os.close(dir_fd)

    def read_registry(self) -> dict[str, Any]:
        """读取注册表，JSON 解码失败时最多重试 3 次。"""
        for attempt in range(3):
            try:
                with open(self._registry_file) as f:
                    return dict(json.load(f))
            except json.JSONDecodeError:
                if attempt < 2:
                    logger.warning(
                        "WeightSharing: registry JSON decode failed, "
                        "retry %d/3 ...", attempt + 1)
                    time.sleep(0.5)
                else:
                    raise

    def registry_exists(self) -> bool:
        """注册表文件是否存在。"""
        return self._registry_file.exists()

    # ------------------------------------------------------------------
    # Primary 存活锁（flock）
    # ------------------------------------------------------------------

    def acquire_primary_lock(self) -> None:
        """获取 primary.lock 的排他锁（LOCK_EX + LOCK_NB）。

        必须在写入注册表之前调用。
        如果另一个 primary 已持有锁，抛出 RuntimeError。
        内核在进程退出时自动释放锁（含 SIGKILL/crash），无需手动释放。
        """
        self.ensure_directory()
        lock_path = str(self._lock_file)
        fd = open(lock_path, "w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fd.close()
            raise RuntimeError(
                f"WeightSharing: 另一个 primary 正持有锁 {lock_path}。"
                f"同一节点同一模型只允许一个 primary。"
            ) from exc
        fd.write(str(os.getpid()))    # 记录 PID 方便调试
        fd.flush()
        # 保持 fd 打开——进程退出时内核自动释放 flock
        self._lock_fd = fd
        logger.debug("WeightSharing: 获取 primary 锁成功 %s", lock_path)

    def is_primary_alive(self) -> bool:
        """判断 primary 进程是否存活。

        跨 K8s pod / PID namespace 也有效，只依赖 flock，不用 /proc/<pid>。
        同时能检测之前 primary 留下的过期 signal 文件。
        """
        if not self._lock_file.exists():
            return False
        try:
            fd = open(self._lock_file, "r")
            try:
                # 非阻塞尝试获取排他锁
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 成功拿到 → 无人持有 → primary 已死
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            except BlockingIOError:
                # 被阻塞 → primary 持有 → primary 存活
                return True
            finally:
                fd.close()
        except OSError:
            # 永久错误（EACCES, ENOENT）→ 无法信任 IPC handle → 假设已死
            return False

    # ------------------------------------------------------------------
    # 就绪信号
    # ------------------------------------------------------------------

    def signal_ready(self) -> None:
        """创建 ready signal 文件（获取锁后调用）。

        先删除上次 primary 实例残留的旧文件（如存在）。
        """
        self.ensure_directory()
        self._ready_file.unlink(missing_ok=True)  # 删除旧信号
        self._ready_file.touch()

    def wait_for_primary(self, timeout_seconds: int = 300) -> bool:
        """阻塞等待 primary 就绪信号出现，并验证 primary 存活。

        两阶段检查：
          阶段一：轮询等待 ready.signal 文件出现（每 0.5s 检查一次）
          阶段二：文件存在后，二次确认 primary 仍存活（避免读到过期文件）

        返回 False 表示超时或 primary 已死。
        """
        deadline = time.monotonic() + timeout_seconds

        # 阶段一：等待 ready signal 出现
        while not self._ready_file.exists():
            if time.monotonic() >= deadline:
                logger.error(
                    "WeightSharing: 等待 primary 进程超时（%d 秒）。",
                    timeout_seconds)
                return False
            if self._lock_file.exists() and not self.is_primary_alive():
                logger.error(
                    "WeightSharing: primary 进程在发出 ready 信号前已死亡。")
                return False
            time.sleep(0.5)

        # 阶段二：ready 文件存在但要确认 primary 真的还在运行
        if not self.is_primary_alive():
            logger.error(
                "WeightSharing: 发现过期 ready 信号 %s "
                "(primary 已退出)。请清理 %s 并重启 primary。",
                self._ready_file, self._base_path)
            return False

        return True

    def cleanup(self) -> None:
        """释放 flock 并删除所有注册表文件。"""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except OSError:
                pass
            self._lock_fd = None

        for p in [self._ready_file, self._registry_file, self._lock_file]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # IPC handle 序列化/反序列化
    # ------------------------------------------------------------------
    # CUDA IPC handle 是 torch 内部对象，不能用 JSON 直接序列化。
    # 先 pickle 再 base64 编码，转为纯文本 JSON 可存储。

    @staticmethod
    def serialize_ipc_handle(handle: IpcHandle) -> str:
        """Pickle + base64 编码 CUDA IPC handle tuple。"""
        return base64.b64encode(pickle.dumps(handle)).decode("utf-8")

    @staticmethod
    def deserialize_ipc_handle(b64_str: str) -> IpcHandle:
        """Base64 + pickle 解码，还原 CUDA IPC handle。"""
        return pickle.loads(base64.b64decode(b64_str))
