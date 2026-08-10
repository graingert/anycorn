"""Tests for the WSGIMiddleware ASGI-to-WSGI adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anyio
import pytest

from anycorn.middleware.wsgi import WSGIMiddleware
from anycorn.typing import ConnectionState

if TYPE_CHECKING:
    from collections.abc import Callable

    from anycorn.typing import (
        ASGIReceiveCallable,
        ASGISendEvent,
        HTTPScope,
    )


def _echo(environ: dict, start_response: Callable) -> list[bytes]:
    body = environ["wsgi.input"].read()
    start_response("200 OK", [("content-type", "text/plain"), ("content-length", str(len(body)))])
    return [body]


@pytest.mark.anyio
async def test_wsgi_middleware_dispatches_to_the_wsgi_app() -> None:
    """The middleware runs the WSGI app in a worker thread and streams back its response."""
    middleware = WSGIMiddleware(_echo)
    assert middleware.max_body_size == 2**16  # the default carried onto the instance

    scope: HTTPScope = {
        "http_version": "1.1",
        "asgi": {},
        "method": "POST",
        "headers": [],
        "path": "/",
        "root_path": "/",
        "query_string": b"",
        "raw_path": b"/",
        "scheme": "http",
        "type": "http",
        "client": ("localhost", 80),
        "server": None,
        "extensions": {},
        "state": ConnectionState({}),
    }
    send_channel, receive_channel = anyio.create_memory_object_stream[dict](1)
    messages: list[ASGISendEvent] = []

    async def _send(message: ASGISendEvent) -> None:
        messages.append(message)

    async with send_channel, receive_channel:
        await send_channel.send({"type": "http.request", "body": b"ping"})
        receive = cast("ASGIReceiveCallable", receive_channel.receive)
        await middleware(scope, receive, _send)

    start = messages[0]
    assert start["type"] == "http.response.start"
    assert start["status"] == 200  # noqa: PLR2004
    body = b"".join(m["body"] for m in messages if m["type"] == "http.response.body")
    assert body == b"ping"
