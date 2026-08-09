"""Tests for the socket-backed TLS stream and listener that back kTLS zero-copy send.

kTLS itself needs a Linux kernel with the ``tls`` ULP loaded and an OpenSSL built with
kTLS, which a restricted sandbox does not provide, so these tests exercise the parts that
are always reachable: the KTLSStream unit tests drive a real ``SSLSocket`` non-blockingly
over a socketpair as ordinary userspace TLS, and the listener tests serve HTTPS through
worker_serve (which uses the KTLSListener where kTLS is available). Where kTLS is not
active, :attr:`KTLSStream.sendfile_socket` is ``None`` and the zerocopysend extension is
not offered, so a plaintext ``sendfile`` is never done over a still-encrypting connection.
"""

from __future__ import annotations

import socket
import ssl
import sys
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
import httpx2
import pytest
import trustme
from anyio.abc import SocketAttribute
from anyio.streams.tls import TLSAttribute

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.ktls import KTLSAttribute, KTLSStream, can_enable_ktls, enable_ktls
from anycorn.run import worker_serve

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import TLSCerts

# kTLS is Linux-only, and the socket-backed KTLSStream drives a non-blocking SSLSocket via
# anyio.wait_readable/writable, which the asyncio proactor loop on Windows does not support.
# Off Linux, can_enable_ktls is False and worker_serve never selects the KTLSListener anyway.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="kTLS and its socket-backed TLS stream are Linux-only"
)


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


async def _serve_tls_and_get(app: Any, tls_certs: TLSCerts) -> httpx2.Response:  # noqa: ANN401
    """Serve *app* over HTTPS with worker_serve (use_ktls) and GET once over TCP.

    With ``use_ktls`` set, worker_serve serves through the KTLSListener wherever kTLS is
    available (Linux with a kTLS OpenSSL) and falls back to ordinary TLS elsewhere; either
    way, without the kernel tls ULP the send path stays in userspace.
    """
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    config.alpn_protocols = ["http/1.1"]
    config.use_ktls = True
    config.accesslog = "-"
    config.errorlog = "-"
    shutdown = anyio.Event()
    verify = ssl.create_default_context(cafile=str(tls_certs.cafile))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as task_group:
            binds: list[str] = await task_group.start(
                lambda *, task_status: worker_serve(
                    ASGIWrapper(app),
                    config,
                    shutdown_trigger=shutdown.wait,
                    task_status=task_status,
                )
            )
            async with httpx2.AsyncClient(base_url=binds[0], verify=verify) as client:
                response = await client.get("/")
            shutdown.set()
    return response


@pytest.mark.anyio
async def test_ktls_listener_serves_https(tls_certs: TLSCerts) -> None:
    """worker_serve with use_ktls serves a whole HTTPS request; here as userspace TLS.

    Without the kernel tls ULP kTLS never activates, so the body is read and encrypted
    through the stream and the zerocopysend extension is deliberately not offered.
    """
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

    response = await _serve_tls_and_get(app, tls_certs)
    response.raise_for_status()
    assert response.content == payload
    assert "tls" in captured["extensions"]
    assert captured["scheme"] == "https"
    # Userspace TLS: a plaintext sendfile would leak, so zerocopysend must stay unadvertised.
    assert "http.response.zerocopysend" not in captured["extensions"]


@pytest.mark.anyio
async def test_ktls_listener_pathsend_falls_back(tmp_path: Path, tls_certs: TLSCerts) -> None:
    """Path send over a userspace-TLS connection reads and encrypts the file via the stream."""
    payload = bytes(range(256)) * 200
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.pathsend", "path": str(file_path)})

    response = await _serve_tls_and_get(app, tls_certs)
    response.raise_for_status()
    assert response.content == payload
