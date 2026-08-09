"""Full-stack zero-copy tests: a real request over a loopback TCP connection, with sendfile.

These drive a whole ``TCPServer`` - HTTP/1.1 parsing, the app, h11 framing and the
os.sendfile in ``_transmit_file`` - over a loopback TCP connection, so the bytes a client
actually receives are asserted end to end. TCP rather than an ``AF_UNIX`` socketpair
because ``os.sendfile`` does not support the latter on macOS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
import pytest
from anyio.abc import SocketStream

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.sendfile import have_sendfile
from anycorn.tcp_server import TCPServer
from anycorn.worker_context import WorkerContext

from .helpers import tcp_socket_pair

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(not have_sendfile, reason="os.sendfile unavailable")


async def _read_response(stream: SocketStream, body_length: int) -> tuple[bytes, bytes]:
    """Read one HTTP response, returning (head, body) once body_length bytes are in."""
    buffer = bytearray()
    with anyio.fail_after(5):
        while b"\r\n\r\n" not in buffer:
            buffer.extend(await stream.receive())
        head, _, body = bytes(buffer).partition(b"\r\n\r\n")
        while len(body) < body_length:
            body += await stream.receive()
    return head, body


async def _serve_one(app: Any, request: bytes, body_length: int) -> tuple[bytes, bytes]:  # noqa: ANN401
    """Serve *app* to a single *request* over a TCP connection and return its (head, body)."""
    server_sock, client_sock = tcp_socket_pair()
    server_stream = await SocketStream.from_socket(server_sock)
    client_stream = await SocketStream.from_socket(client_sock)
    server = TCPServer(ASGIWrapper(app), Config(), WorkerContext(None), {}, server_stream)

    head = body = b""
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(server.run)
        await client_stream.send(request)
        head, body = await _read_response(client_stream, body_length)
        task_group.cancel_scope.cancel()
    await client_stream.aclose()
    return head, body


@pytest.mark.anyio
async def test_pathsend_delivers_the_file_over_a_socket(tmp_path: Path) -> None:
    """Path send transmits the whole named file as the response body via os.sendfile."""
    payload = b"pathsend zero copy\n" * 10_000
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.pathsend", "path": str(file_path)})

    head, body = await _serve_one(app, b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n", len(payload))
    assert head.startswith(b"HTTP/1.1 200")
    assert body == payload


@pytest.mark.anyio
async def test_zerocopysend_delivers_a_file_descriptor_over_a_socket(tmp_path: Path) -> None:
    """The app hands over an open descriptor and offset/count; the window is sent."""
    payload = bytes(range(256)) * 500
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    window = payload[100:-100]

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        assert "http.response.zerocopysend" in scope["extensions"]
        file = file_path.open("rb")
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", str(len(window)).encode())],
                }
            )
            await send(
                {
                    "type": "http.response.zerocopysend",
                    "file": file,
                    "offset": 100,
                    "count": len(window),
                }
            )
        finally:
            file.close()

    head, body = await _serve_one(app, b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n", len(window))
    assert head.startswith(b"HTTP/1.1 200")
    assert body == window
