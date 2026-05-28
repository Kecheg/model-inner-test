#!/usr/bin/env python3
"""Launch vLLM with an explicit kvcached autopatch bootstrap.

Sardeenz previously relied on kvcached's .pth-based startup hook to import
the vLLM autopatch module. In editable installs that hook may not be installed,
which causes models to run without kvcached even though the environment flags
are set. Importing the autopatch module here makes the bootstrap explicit and
fail-closed for kvcached-enabled launches.
"""

from __future__ import annotations

import os
import inspect
import sys

from fastapi import Request
from fastapi.responses import JSONResponse


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "false").lower() in ("true", "1")


def _install_kvcached_trim_route() -> None:
    import vllm.entrypoints.openai.api_server as api_server

    if getattr(api_server.build_app, "_sardeenz_kvcached_trim_patched", False):
        return

    original_build_app = api_server.build_app

    def _call_trim(engine_client):
        if hasattr(engine_client, "call_utility_async"):
            return engine_client.call_utility_async("trim_kvcached")
        if hasattr(engine_client, "call_utility"):
            return engine_client.call_utility("trim_kvcached")
        engine_core = getattr(engine_client, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "call_utility_async"):
            return engine_core.call_utility_async("trim_kvcached")
        if engine_core is not None and hasattr(engine_core, "call_utility"):
            return engine_core.call_utility("trim_kvcached")
        if engine_core is not None and hasattr(engine_core, "trim_kvcached"):
            return engine_core.trim_kvcached()
        raise RuntimeError("engine client does not expose a trim_kvcached utility path")

    def patched_build_app(*args, **kwargs):
        app = original_build_app(*args, **kwargs)

        @app.post("/kvcached/trim")
        async def kvcached_trim(raw_request: Request):
            result = _call_trim(raw_request.app.state.engine_client)
            if inspect.isawaitable(result):
                result = await result
            return JSONResponse(content={"ok": True, "result": result})

        return app

    patched_build_app._sardeenz_kvcached_trim_patched = True
    api_server.build_app = patched_build_app


if _env_enabled("ENABLE_KVCACHED") and _env_enabled("KVCACHED_AUTOPATCH"):
    try:
        import kvcached.integration.vllm.autopatch  # noqa: F401
        _install_kvcached_trim_route()
    except Exception as exc:  # pragma: no cover - startup guard
        print(f"Failed to initialize kvcached autopatch: {exc}", file=sys.stderr)
        raise

from vllm.entrypoints.cli.main import main


if __name__ == "__main__":
    sys.exit(main())
