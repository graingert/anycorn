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
from cryptography import x509

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.ktls import (
    KTLSAttribute,
    KTLSListener,
    KTLSStream,
    _ktls_send_active,
    can_enable_ktls,
    enable_ktls,
)
from anycorn.run import worker_serve
from anycorn.utils import build_tls_extension

if TYPE_CHECKING:
    from pathlib import Path

    from anycorn.typing import TLSExtension
    from tests.conftest import TLSCerts

# kTLS is Linux-only, and the socket-backed KTLSStream drives a non-blocking SSLSocket via
# anyio.wait_readable/writable, which the asyncio proactor loop on Windows does not support.
# Off Linux, can_enable_ktls is False and worker_serve never selects the KTLSListener anyway.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="kTLS and its socket-backed TLS stream are Linux-only"
)

TLS_VERSION_1_3 = 0x0304  # RFC 8446
TLS_1_3_CIPHER_SUITES = frozenset({0x1301, 0x1302, 0x1303})  # AES-128/256-GCM, ChaCha20


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


@pytest.mark.anyio
async def test_tls_extension_is_populated_over_ktls(
    tmp_path: Path, certificate_authority: trustme.CA
) -> None:
    """build_tls_extension harvests version, cipher and the client cert from a KTLSStream.

    KTLSStream exposes an ssl.SSLSocket rather than the ssl.SSLObject an in-memory TLS stream
    gives, so this guards that the harvester still reads every field off it. The extension
    reflects the negotiated session, not the send path, so it is the same whether or not kTLS
    activates - which lets this run as userspace TLS in a sandbox without the kernel tls ULP.
    """
    server_cert = certificate_authority.issue_cert("localhost")
    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    server_cert.cert_chain_pems[0].write_to_path(str(certfile))
    server_cert.private_key_pem.write_to_path(str(keyfile))

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(certfile), str(keyfile))
    server_ctx.verify_mode = ssl.CERT_REQUIRED  # ask the client for a certificate
    certificate_authority.configure_trust(server_ctx)
    enable_ktls(server_ctx)

    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    certificate_authority.configure_trust(client_ctx)
    client_cert = certificate_authority.issue_cert("client@example.com")
    client_cert.configure_cert(client_ctx)
    client_ctx.minimum_version = ssl.TLSVersion.TLSv1_3  # so the negotiated version is exact

    server_sock, client_sock = socket.socketpair()
    config = Config()
    config.certfile = str(certfile)
    extension: TLSExtension | None = None

    async def run_server() -> None:
        nonlocal extension
        stream = await KTLSStream.wrap(server_sock, ssl_context=server_ctx)
        extension = build_tls_extension(config, stream)
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

    assert extension is not None
    server_leaf = ssl.PEM_cert_to_DER_cert(server_cert.cert_chain_pems[0].bytes().decode())
    client_leaf = ssl.PEM_cert_to_DER_cert(client_cert.cert_chain_pems[0].bytes().decode())
    issuing_ca = ssl.PEM_cert_to_DER_cert(certificate_authority.cert_pem.bytes().decode())
    expected_client_name = x509.load_pem_x509_certificate(
        client_cert.cert_chain_pems[0].bytes()
    ).subject.rfc4514_string()

    assert extension["tls_version"] == TLS_VERSION_1_3
    assert extension["cipher_suite"] in TLS_1_3_CIPHER_SUITES
    server_cert_pem = extension["server_cert"]
    assert server_cert_pem is not None
    assert ssl.PEM_cert_to_DER_cert(server_cert_pem) == server_leaf
    # The client presented a certificate the server trusts, so it is harvested with no error.
    assert extension["client_cert_error"] is None
    assert extension["client_cert_name"] == expected_client_name
    # The harvested chain is exactly the certificates the client presented, compared by DER so
    # PEM formatting does not matter. get_verified_chain (CPython 3.13+) returns the client leaf
    # followed by the issuing CA; older Pythons' getpeercert fallback returns the leaf alone.
    chain = [ssl.PEM_cert_to_DER_cert(pem) for pem in extension["client_cert_chain"]]
    assert chain == ([client_leaf, issuing_ca] if sys.version_info >= (3, 13) else [client_leaf])


def test_ktls_send_active_short_circuits_without_the_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the capability flag off, the send probe returns False without touching the socket.

    On a host that really has kTLS the flag is otherwise always on, so forcing it off is what
    exercises this guard there as well as on hosts that never had kTLS.
    """
    monkeypatch.setattr("anycorn.ktls.can_enable_ktls", False)
    sock = socket.socket()
    with sock:
        assert _ktls_send_active(sock) is False


def test_can_enable_ktls_matches_platform() -> None:
    """The capability flag is exactly Linux plus an OpenSSL that exposes kTLS.

    This module only runs on Linux (see the skipif), so the flag tracks the OpenSSL constant;
    written without a branch so it is covered whether or not this build actually has kTLS.
    """
    assert can_enable_ktls == hasattr(ssl, "OP_ENABLE_KTLS")


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
    """worker_serve with use_ktls serves a whole HTTPS request over the KTLSListener.

    Whether kTLS actually activates (and so whether zerocopysend is offered) depends on the
    kernel tls ULP being available, which differs between machines, so this asserts only what
    holds either way - the response is served correctly over TLS. The kTLS-active case, where
    zerocopysend is offered and the body goes out with os.sendfile, is covered by
    tests/e2e/test_ktls_real.py.
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


@pytest.mark.anyio
async def test_ktls_listener_serves_pathsend(tmp_path: Path, tls_certs: TLSCerts) -> None:
    """Path send is delivered correctly over TLS, whether by kTLS os.sendfile or userspace."""
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


@pytest.fixture
def _force_ktls_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force can_enable_ktls on so the kTLS-guarded paths are walked without a real ULP.

    The sandbox kernel has no tls ULP, so _ktls_send_active's getsockopt still fails and the
    send path stays userspace - but the code past the ``if not can_enable_ktls`` guard runs,
    which is otherwise reachable only on a real kTLS host.
    """
    monkeypatch.setattr("anycorn.ktls.can_enable_ktls", True)


def _bound_tcp_listener() -> tuple[socket.socket, str, int]:
    """Return a listening IPv4 loopback socket, plus the host and port it is bound to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    host, port = sock.getsockname()
    return sock, host, port


@pytest.mark.anyio
@pytest.mark.usefixtures("_force_ktls_available")
async def test_ktls_listener_accepts_and_handles_a_connection(
    certificate_authority: trustme.CA,
) -> None:
    """The listener accepts a real TCP connection, handshakes it and runs the handler.

    This drives serve/_accept/_handshake_and_handle and the kTLS probe in KTLSStream.wrap end
    to end as userspace TLS, the whole path a real kTLS host would take bar the kernel send.
    """
    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)
    listen_sock, host, port = _bound_tcp_listener()
    listener = KTLSListener(listen_sock, server_ctx, handshake_timeout=5)

    received = b""
    handled = anyio.Event()

    async def handler(stream: KTLSStream) -> None:
        nonlocal received
        received = await stream.receive()
        await stream.send(b"pong")
        await stream.aclose()
        handled.set()

    def talk() -> bytes:
        raw = socket.create_connection((host, port))
        tls = client_ctx.wrap_socket(raw, server_hostname="localhost")
        tls.sendall(b"ping")
        data = tls.recv(4)
        tls.close()
        return data

    with anyio.fail_after(10):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(listener.serve, handler)
            client_got = await anyio.to_thread.run_sync(talk)
            await handled.wait()
            task_group.cancel_scope.cancel()

    assert listener.extra_attributes[SocketAttribute.family]() == socket.AF_INET
    assert received == b"ping"
    assert client_got == b"pong"
    await listener.aclose()


@pytest.mark.anyio
@pytest.mark.usefixtures("_force_ktls_available")
async def test_ktls_listener_survives_a_failed_handshake(
    certificate_authority: trustme.CA,
) -> None:
    """A client that sends garbage is dropped without the handler running or serve stopping.

    The bad handshake makes OpenSSL raise an ``ssl.SSLError`` (an ``OSError``), which _retry
    maps to ``BrokenResourceError``; _handshake_and_handle logs it and closes the connection.
    """
    server_ctx = _server_context(certificate_authority)
    listen_sock, host, port = _bound_tcp_listener()
    listener = KTLSListener(listen_sock, server_ctx, handshake_timeout=5)

    async def handler(stream: KTLSStream) -> None:  # noqa: ARG001
        pytest.fail("the handler must not run when the handshake fails")  # pragma: no cover

    def talk() -> None:
        raw = socket.create_connection((host, port))
        raw.sendall(b"this is not a TLS ClientHello\n")
        # Close at once: the server reads the garbage, wants more, then sees EOF and fails
        # the handshake - rather than both sides blocking waiting for the other.
        raw.close()

    with anyio.fail_after(10):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(listener.serve, handler)
            await anyio.to_thread.run_sync(talk)
            await anyio.sleep(0.3)  # let the handshake task finish failing
            task_group.cancel_scope.cancel()

    await listener.aclose()


@pytest.mark.anyio
@pytest.mark.usefixtures("_force_ktls_available")
async def test_ktls_listener_reraises_cancellation_mid_handshake(
    certificate_authority: trustme.CA,
) -> None:
    """Cancelling serve while a handshake is in flight propagates, rather than being swallowed.

    Passing serve an explicit task group also drives the branch where it does not open its own.
    The client connects but never sends a ClientHello, so the handshake blocks until cancelled.
    """
    server_ctx = _server_context(certificate_authority)
    listen_sock, host, port = _bound_tcp_listener()
    listener = KTLSListener(listen_sock, server_ctx)

    async def handler(stream: KTLSStream) -> None:  # noqa: ARG001
        pytest.fail(  # pragma: no cover
            "the handler must not run when the handshake is cancelled"
        )

    raw = socket.create_connection((host, port))
    try:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(listener.serve, handler, task_group)
                await anyio.sleep(0.3)  # let accept run and the handshake start blocking
                task_group.cancel_scope.cancel()
    finally:
        raw.close()
        await listener.aclose()


@pytest.mark.anyio
async def test_receive_ends_the_stream_on_an_empty_read(
    certificate_authority: trustme.CA, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty read - what a peer's clean close_notify produces - ends the stream.

    A real close_notify races the socket teardown across backends, so the empty read the
    handshake-completed stream would see is injected directly, exercising the ``if not data``
    end-of-stream path deterministically.
    """
    server_sock, client_sock = socket.socketpair()
    server_ctx = _server_context(certificate_authority)
    client_ctx = _client_context(certificate_authority)

    async def run_server() -> None:
        stream = await KTLSStream.wrap(server_sock, ssl_context=server_ctx)

        async def empty_read(_func: Any, *_args: Any) -> bytes:  # noqa: ANN401
            return b""

        monkeypatch.setattr(stream, "_retry", empty_read)
        with pytest.raises(anyio.EndOfStream):
            await stream.receive()
        await stream.aclose()

    def talk() -> None:
        client_sock.setblocking(True)  # noqa: FBT003
        tls = client_ctx.wrap_socket(client_sock, server_hostname="localhost")
        tls.close()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_server)
        task_group.start_soon(lambda: anyio.to_thread.run_sync(talk))
