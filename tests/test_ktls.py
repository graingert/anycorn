"""Tests for the socket-backed TLS stream that backs kTLS zero-copy send.

kTLS itself needs a Linux kernel with the ``tls`` ULP loaded and an OpenSSL built with
kTLS, which a restricted sandbox does not provide, so these tests exercise the part that
is always reachable: driving a real ``SSLSocket`` non-blockingly over an ``AF_UNIX``
socketpair - handshake, send, receive and close - as ordinary userspace TLS. Where kTLS is
not active, :attr:`KTLSStream.sendfile_socket` must be ``None`` so callers never do a
plaintext ``sendfile`` over a still-encrypting connection.
"""

from __future__ import annotations

import socket
import ssl
from typing import TYPE_CHECKING, Any

import anyio
import pytest
import trustme
from anyio.abc import SocketAttribute
from anyio.streams.tls import TLSAttribute

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.ktls import KTLSAttribute, KTLSListener, KTLSStream, can_enable_ktls, enable_ktls
from anycorn.tcp_server import tcp_server_handler
from anycorn.worker_context import WorkerContext

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def certificate_authority() -> trustme.CA:
    return trustme.CA()


def _server_context(ca: trustme.CA) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert = ca.issue_cert("localhost")
    cert.configure_cert(context)
    enable_ktls(context)
    return context


def _client_context(ca: trustme.CA) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ca.configure_trust(context)
    return context


@pytest.mark.anyio
async def test_handshake_and_round_trip(certificate_authority: trustme.CA) -> None:
    """A KTLSStream server talks TLS to a stdlib client over a socketpair."""
    server_sock, client_sock = socket.socketpair()
    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)

    received: bytes = b""

    async def run_server() -> None:
        stream = await KTLSStream.wrap(server_sock, ssl_context=server_ctx)
        assert await stream.receive() == b"ping"
        await stream.send(b"pong")
        await stream.aclose()

    async def run_client() -> None:
        nonlocal received
        # The stdlib client side runs in a worker thread with a blocking handshake, so the
        # server's non-blocking KTLSStream is what is under test here.
        def talk() -> bytes:
            client_sock.setblocking(True)  # noqa: FBT003
            tls = client_ctx.wrap_socket(client_sock, server_hostname="localhost")
            tls.sendall(b"ping")
            data = tls.recv(4)
            tls.close()
            return data

        received = await anyio.to_thread.run_sync(talk)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_server)
        task_group.start_soon(run_client)

    assert received == b"pong"


@pytest.mark.anyio
async def test_sendfile_socket_none_without_ktls(certificate_authority: trustme.CA) -> None:
    """When the kernel has not taken over send, no plaintext sendfile socket is exposed."""
    server_sock, client_sock = socket.socketpair()
    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)

    sendfile_socket: object = "unset"
    exposed_attr: object = "unset"

    async def run_server() -> None:
        nonlocal sendfile_socket, exposed_attr
        stream = await KTLSStream.wrap(server_sock, ssl_context=server_ctx)
        sendfile_socket = stream.sendfile_socket
        exposed_attr = stream.extra(KTLSAttribute.sendfile_socket)  # noqa: S610
        await stream.receive()
        await stream.aclose()

    async def run_client() -> None:
        def talk() -> None:
            client_sock.setblocking(True)  # noqa: FBT003
            tls = client_ctx.wrap_socket(client_sock, server_hostname="localhost")
            tls.sendall(b"hi")
            tls.close()

        await anyio.to_thread.run_sync(talk)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_server)
        task_group.start_soon(run_client)

    # The sandbox kernel has no tls ULP, so kTLS never activates and sendfile must be off.
    assert sendfile_socket is None
    assert exposed_attr is None


@pytest.mark.anyio
async def test_exposes_socket_and_tls_attributes(certificate_authority: trustme.CA) -> None:
    """The stream advertises the underlying socket and negotiated TLS parameters."""
    server_sock, client_sock = socket.socketpair()
    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)

    tls_version: str | None = None
    family: int | None = None

    async def run_server() -> None:
        nonlocal tls_version, family
        stream = await KTLSStream.wrap(server_sock, ssl_context=server_ctx)
        tls_version = stream.extra(TLSAttribute.tls_version)  # noqa: S610
        family = stream.extra(SocketAttribute.family)  # noqa: S610
        await stream.receive()
        await stream.aclose()

    async def run_client() -> None:
        def talk() -> None:
            client_sock.setblocking(True)  # noqa: FBT003
            tls = client_ctx.wrap_socket(client_sock, server_hostname="localhost")
            tls.sendall(b"hi")
            tls.close()

        await anyio.to_thread.run_sync(talk)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_server)
        task_group.start_soon(run_client)

    assert tls_version is not None
    assert tls_version.startswith("TLS")
    assert family == socket.AF_UNIX


def test_can_enable_ktls_matches_platform() -> None:
    """The capability flag is only true on Linux with an OpenSSL that exposes kTLS."""
    if can_enable_ktls:
        assert hasattr(ssl, "OP_ENABLE_KTLS")


def _tls_client_request(
    sock_path: str, client_ctx: ssl.SSLContext, request: bytes, body_length: int
) -> tuple[bytes, bytes]:
    """Blocking helper: make one TLS request over an AF_UNIX socket, return (head, body)."""
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.connect(sock_path)
    tls = client_ctx.wrap_socket(raw, server_hostname="localhost")
    try:
        tls.sendall(request)
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            buffer += tls.recv(4096)
        head, _, body = buffer.partition(b"\r\n\r\n")
        while len(body) < body_length:
            body += tls.recv(4096)
    finally:
        tls.close()
    return head, body


@pytest.mark.anyio
async def test_ktls_listener_serves_https(
    tmp_path: Path, certificate_authority: trustme.CA
) -> None:
    """KTLSListener drives a whole HTTPS request/response, falling back to userspace TLS.

    The sandbox has no kernel tls ULP, so kTLS never activates: the response body is read and
    encrypted through the stream, and the zerocopysend extension is deliberately not offered.
    """
    sock_path = str(tmp_path / "https.sock")
    listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen_sock.bind(sock_path)
    listen_sock.listen(1)

    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)
    payload = b"hello over a kTLS-capable listener\n" * 100
    captured: dict[str, Any] = {}

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        captured["scheme"] = scope["scheme"]
        captured["extensions"] = dict(scope["extensions"])
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    handler = tcp_server_handler(ASGIWrapper(app), Config(), WorkerContext(None), {})
    listener = KTLSListener(listen_sock, server_ctx)
    request = b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(listener.serve, handler)
        head, body = await anyio.to_thread.run_sync(
            _tls_client_request, sock_path, client_ctx, request, len(payload)
        )
        task_group.cancel_scope.cancel()
    listen_sock.close()

    assert head.startswith(b"HTTP/1.1 200")
    assert body == payload
    assert captured["scheme"] == "https"
    assert "tls" in captured["extensions"]
    # Userspace TLS: a plaintext sendfile would leak, so zerocopysend must stay unadvertised.
    assert "http.response.zerocopysend" not in captured["extensions"]


@pytest.mark.anyio
async def test_ktls_listener_pathsend_falls_back(
    tmp_path: Path, certificate_authority: trustme.CA
) -> None:
    """Path send over a userspace-TLS KTLSListener reads and encrypts the file via the stream."""
    payload = bytes(range(256)) * 200
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)

    sock_path = str(tmp_path / "https.sock")
    listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen_sock.bind(sock_path)
    listen_sock.listen(1)

    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.pathsend", "path": str(file_path)})

    handler = tcp_server_handler(ASGIWrapper(app), Config(), WorkerContext(None), {})
    listener = KTLSListener(listen_sock, server_ctx)
    request = b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(listener.serve, handler)
        head, body = await anyio.to_thread.run_sync(
            _tls_client_request, sock_path, client_ctx, request, len(payload)
        )
        task_group.cancel_scope.cancel()
    listen_sock.close()

    assert head.startswith(b"HTTP/1.1 200")
    assert body == payload
