"""Real kernel-TLS tests that require the kernel to actually take over the TLS send path.

Unlike ``tests/test_ktls.py``, which exercises the socket-backed TLS stream as userspace TLS
(the only thing a restricted sandbox can do), these use a real TCP connection and require
kTLS to activate: one drives the whole ``worker_serve`` with ``use_ktls`` so an HTTP/1.1 body
really goes out with ``os.sendfile`` while the kernel encrypts it, and one confirms kTLS also
offloads an HTTP/2 connection (where the framed body cannot use zerocopysend).

They are gated on the ``ANYCORN_REQUIRE_KTLS`` environment variable, which GitHub CI sets on a
Linux job that has loaded the kernel ``tls`` module. When selected they must *activate* kTLS:
they fail - rather than skip - if OpenSSL, the platform or the kernel does not provide it, so
a CI environment that quietly lacks working kTLS is caught instead of passing silently.
"""

from __future__ import annotations

import os
import socket
import ssl
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
import httpx2
import pytest
from anyio.streams.tls import TLSAttribute

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.ktls import KTLSStream, can_enable_ktls, enable_ktls
from anycorn.run import worker_serve

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import TLSCerts

REQUIRE_KTLS = os.environ.get("ANYCORN_REQUIRE_KTLS") == "1"

pytestmark = pytest.mark.skipif(
    not REQUIRE_KTLS,
    reason="set ANYCORN_REQUIRE_KTLS=1 (as the GitHub CI kTLS job does) to run the real kTLS test",
)

HOST = "127.0.0.1"


@pytest.mark.anyio
async def test_worker_serve_uses_real_ktls_for_zero_copy_send(
    tls_certs: TLSCerts, free_tcp_port: int, tmp_path: Path
) -> None:
    """worker_serve activates kTLS so a TLS body is sent with os.sendfile.

    This selects the whole path: TLS handshake on a socket-backed stream, kTLS negotiation,
    the zerocopysend ASGI extension, and the os.sendfile in the TCP server - and proves kTLS
    is live via the one signal that is true only when the kernel owns the TLS send path: the
    server offering ``http.response.zerocopysend`` on a TLS connection.
    """
    # Selected only where kTLS must work, so fail loudly instead of skipping.
    assert can_enable_ktls, (
        "OpenSSL/platform does not expose kTLS (ssl.OP_ENABLE_KTLS missing or non-Linux); "
        "this test requires a kTLS-capable OpenSSL on Linux"
    )

    payload = bytes(range(256)) * 4000  # ~1 MiB, well past a single TLS record
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)

    seen: dict[str, Any] = {}

    config = Config()
    config.bind = [f"{HOST}:{free_tcp_port}"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    config.alpn_protocols = ["http/1.1"]
    config.use_ktls = True
    config.accesslog = "-"
    config.errorlog = "-"

    shutdown = anyio.Event()
    verify = ssl.create_default_context(cafile=str(tls_certs.cafile))

    # Opened here, not inside the app: the app runs in the server's task group, which is
    # cancelled at shutdown, and an awaited close racing that cancellation can leak the file.
    file = await anyio.Path(file_path).open("rb")
    try:

        async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
            seen["extensions"] = dict(scope["extensions"])
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", str(len(payload)).encode())],
                }
            )
            await send(
                {
                    "type": "http.response.zerocopysend",
                    "file": file,
                    "offset": 0,
                    "count": len(payload),
                }
            )

        with anyio.fail_after(30):
            async with anyio.create_task_group() as task_group:
                await task_group.start(
                    lambda *, task_status: worker_serve(
                        ASGIWrapper(app),
                        config,
                        shutdown_trigger=shutdown.wait,
                        task_status=task_status,
                    )
                )
                async with httpx2.AsyncClient(
                    base_url=f"https://{HOST}:{free_tcp_port}", verify=verify
                ) as client:
                    response = await client.get("/")
                shutdown.set()
    finally:
        await file.aclose()

    response.raise_for_status()
    assert response.content == payload
    # The decisive kTLS proof: zerocopysend is advertised on a TLS connection only when the
    # kernel has taken over the send path. Userspace TLS would omit it, failing this test.
    assert "http.response.zerocopysend" in seen["extensions"], (
        "kTLS did not activate: this TLS connection was served in userspace, so zero-copy "
        "send was not offered. Load the kernel 'tls' module (modprobe tls) and ensure the "
        "negotiated cipher is kTLS-compatible."
    )


@pytest.mark.anyio
async def test_ktls_activates_on_an_http2_connection(tls_certs: TLSCerts) -> None:
    """Kernel TLS offloads the send path on an ALPN-negotiated HTTP/2 connection.

    kTLS lives at the TLS record layer, below ALPN, so it applies to HTTP/2 - whose framed
    body cannot use zerocopysend - just as to HTTP/1.1: the kernel still encrypts the h2
    frames. This negotiates "h2" over a real TCP handshake and asserts the kernel took over
    the send path, which KTLSStream reports only when kTLS TX is active.
    """
    assert can_enable_ktls, (
        "OpenSSL/platform does not expose kTLS (ssl.OP_ENABLE_KTLS missing or non-Linux); "
        "this test requires a kTLS-capable OpenSSL on Linux"
    )

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(tls_certs.certfile), str(tls_certs.keyfile))
    server_ctx.set_alpn_protocols(["h2", "http/1.1"])
    enable_ktls(server_ctx)
    client_ctx = ssl.create_default_context(cafile=str(tls_certs.cafile))
    client_ctx.set_alpn_protocols(["h2"])

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((HOST, 0))
    listener.listen(1)
    listener.setblocking(False)  # noqa: FBT003
    port = listener.getsockname()[1]

    server_alpn: str | None = None
    sendfile_socket: object = "unset"

    async def run_server() -> None:
        nonlocal server_alpn, sendfile_socket
        while True:
            try:
                conn, _ = listener.accept()
            except BlockingIOError:  # noqa: PERF203
                await anyio.wait_readable(listener)
            else:
                break
        stream = await KTLSStream.wrap(conn, ssl_context=server_ctx)
        server_alpn = stream.extra(TLSAttribute.alpn_protocol)  # noqa: S610
        sendfile_socket = stream.sendfile_socket
        await stream.send(b"ping")
        await stream.aclose()

    client_alpn: str | None = None

    def run_client() -> None:
        nonlocal client_alpn
        with socket.create_connection((HOST, port)) as raw:
            tls = client_ctx.wrap_socket(raw, server_hostname="localhost")
            try:
                client_alpn = tls.selected_alpn_protocol()
                tls.recv(4)
            finally:
                tls.close()

    with anyio.fail_after(30):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_server)
            await anyio.to_thread.run_sync(run_client)

    listener.close()

    assert client_alpn == "h2"
    assert server_alpn == "h2"
    assert sendfile_socket is not None, (
        "kTLS did not activate on the HTTP/2 connection: the kernel did not take over the "
        "TLS send path (KTLSStream.sendfile_socket is None)."
    )
