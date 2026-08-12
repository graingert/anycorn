"""A minimal ASGI app for the run() tests, loadable by a real worker process."""

from __future__ import annotations

from typing import Any


async def app(_scope: Any, _receive: Any, send: Any) -> None:  # noqa: ANN401
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})
