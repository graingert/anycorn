"""A real end-to-end kernel-TLS test: worker_serve over TLS with os.sendfile via kTLS.

Unlike ``tests/test_ktls.py``, which exercises the socket-backed TLS stream as userspace TLS
(the only thing a restricted sandbox can do), this drives the real ``worker_serve`` over a
real TCP socket with ``use_ktls`` and requires the kernel to actually take over the TLS send
path, so the response body really goes out with ``os.sendfile`` while the kernel encrypts it.

It is gated on the ``ANYCORN_REQUIRE_KTLS`` environment variable, which GitHub CI sets on a
Linux job that has loaded the kernel ``tls`` module. When selected it must *activate* kTLS:
it fails - rather than skips - if OpenSSL, the platform or the kernel does not provide it, so
a CI environment that quietly lacks working kTLS is caught instead of passing silently.
"""

from __future__ import annotations

import os
import ssl
from typing import TYPE_CHECKING, Any

import anyio
import httpx2
import pytest

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.ktls import can_enable_ktls
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
TLS_HOST = "localhost"


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
    file_path.write_bytes(payload)

    seen: dict[str, Any] = {}

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        seen["extensions"] = dict(scope["extensions"])
        file = file_path.open("rb")
        try:
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
        finally:
            file.close()

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
                base_url=f"https://{TLS_HOST}:{free_tcp_port}", verify=verify
            ) as client:
                response = await client.get("/")
            shutdown.set()

    response.raise_for_status()
    assert response.content == payload
    # The decisive kTLS proof: zerocopysend is advertised on a TLS connection only when the
    # kernel has taken over the send path. Userspace TLS would omit it, failing this test.
    assert "http.response.zerocopysend" in seen["extensions"], (
        "kTLS did not activate: this TLS connection was served in userspace, so zero-copy "
        "send was not offered. Load the kernel 'tls' module (modprobe tls) and ensure the "
        "negotiated cipher is kTLS-compatible."
    )
