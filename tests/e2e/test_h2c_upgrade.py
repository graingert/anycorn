"""End-to-end test of the HTTP/1.1 to HTTP/2 cleartext (h2c) Upgrade mechanism.

No mainstream client library emits an h2c Upgrade, so the client side is built here from
the ``h2`` library over a raw socket. The server is a real ``anycorn.serve`` instance, so
the whole upgrade - the 101 Switching Protocols handshake, the switch to the HTTP/2 handler,
and the response to the upgraded request - runs against the real protocol stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
import h2.config
import h2.connection
import h2.events
import pytest

import anycorn
from anycorn.config import Config

if TYPE_CHECKING:
    from anyio.abc import SocketStream

HOST = "127.0.0.1"


async def _app(scope: Any, _receive: Any, send: Any) -> None:  # noqa: ANN401
    assert scope["type"] == "http"
    assert scope["http_version"] == "2"  # the upgraded request is served over HTTP/2
    await send(
        {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"5")]}
    )
    await send({"type": "http.response.body", "body": b"hello"})


async def _read_101_and_remainder(stream: SocketStream) -> bytes:
    """Read the 101 Switching Protocols response, returning any HTTP/2 bytes that follow it."""
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        buffer += await stream.receive()
    head, remainder = buffer.split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 101"), head
    return remainder


@pytest.mark.anyio
async def test_h2c_cleartext_upgrade_serves_the_request_over_http2(free_tcp_port: int) -> None:
    config = Config()
    config.bind = [f"{HOST}:{free_tcp_port}"]
    config.accesslog = "-"
    config.errorlog = "-"

    connection = h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=True))
    settings_header = connection.initiate_upgrade_connection()  # sets up stream 1 as the request
    assert settings_header is not None

    async with anyio.create_task_group() as task_group:
        shutdown = anyio.Event()
        await task_group.start(
            lambda *, task_status: anycorn.serve(
                _app, config, shutdown_trigger=shutdown.wait, task_status=task_status
            )
        )

        async with await anyio.connect_tcp(HOST, free_tcp_port) as stream:
            request = (
                b"GET / HTTP/1.1\r\n"
                b"host: " + HOST.encode() + b"\r\n"
                b"connection: Upgrade, HTTP2-Settings\r\n"
                b"upgrade: h2c\r\n"
                b"http2-settings: " + settings_header + b"\r\n\r\n"
            )
            await stream.send(request)
            await stream.send(connection.data_to_send())  # client preface + SETTINGS

            status: bytes | None = None
            body = b""
            ended = False
            data = await _read_101_and_remainder(stream)
            with anyio.fail_after(10):
                while not ended:
                    for event in connection.receive_data(data):
                        if isinstance(event, h2.events.ResponseReceived):
                            status = dict(event.headers)[b":status"]
                        elif isinstance(event, h2.events.DataReceived):
                            body += event.data
                        elif isinstance(event, h2.events.StreamEnded):
                            ended = True
                    if to_send := connection.data_to_send():
                        await stream.send(to_send)
                    if not ended:
                        data = await stream.receive()

        shutdown.set()

    assert status == b"200"
    assert body == b"hello"
