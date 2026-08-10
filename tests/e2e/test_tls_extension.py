"""End-to-end test that worker_serve builds the ASGI TLS extension over a real connection.

The unit tests in ``tests/test_utils.py`` feed ``build_tls_extension`` a hand-written fake ssl
object. This drives the real ``worker_serve`` over a real (non-kTLS) TLS connection with mutual
TLS and asserts the ``tls`` extension in the ASGI scope holds the exact certificates the client
presented - the counterpart, on the ordinary TLS path, to the KTLSStream test in
``tests/test_ktls.py``.
"""

from __future__ import annotations

import ssl
import sys
from typing import TYPE_CHECKING, Any

import anyio
import httpx2
import pytest
import trustme
from cryptography import x509

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.run import worker_serve

if TYPE_CHECKING:
    from pathlib import Path

HOST = "127.0.0.1"
TLS_VERSION_1_3 = 0x0304  # RFC 8446
TLS_1_3_CIPHER_SUITES = frozenset({0x1301, 0x1302, 0x1303})  # AES-128/256-GCM, ChaCha20


@pytest.mark.anyio
async def test_worker_serve_builds_the_tls_extension(tmp_path: Path) -> None:  # noqa: PLR0915
    """The tls extension in the scope holds the client certificate harvested over real TLS."""
    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost", HOST)
    client_cert = ca.issue_cert("client@example.com")
    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    cafile = tmp_path / "ca.pem"
    server_cert.cert_chain_pems[0].write_to_path(str(certfile))
    server_cert.private_key_pem.write_to_path(str(keyfile))
    ca.cert_pem.write_to_path(str(cafile))
    client_certfile = tmp_path / "client.pem"
    client_keyfile = tmp_path / "client_key.pem"
    client_cert.cert_chain_pems[0].write_to_path(str(client_certfile))
    client_cert.private_key_pem.write_to_path(str(client_keyfile))

    config = Config()
    config.bind = [f"{HOST}:0"]  # OS-assigned port; the bound URL comes back via task_status
    config.certfile = str(certfile)
    config.keyfile = str(keyfile)
    config.ca_certs = str(cafile)
    config.verify_mode = ssl.CERT_REQUIRED  # require and verify a client certificate
    config.alpn_protocols = ["http/1.1"]
    config.accesslog = "-"
    config.errorlog = "-"

    captured: dict[str, Any] = {}

    async def app(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401, ARG001
        captured["tls"] = dict(scope["extensions"].get("tls", {}))
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
        )
        await send({"type": "http.response.body", "body": b"hi"})

    client_ctx = ssl.create_default_context(cafile=str(cafile))
    client_ctx.load_cert_chain(str(client_certfile), str(client_keyfile))
    client_ctx.minimum_version = ssl.TLSVersion.TLSv1_3  # so the negotiated version is exact
    shutdown = anyio.Event()

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
            async with httpx2.AsyncClient(base_url=binds[0], verify=client_ctx) as client:
                response = await client.get("/")
            shutdown.set()

    response.raise_for_status()
    tls = captured["tls"]

    server_leaf = ssl.PEM_cert_to_DER_cert(server_cert.cert_chain_pems[0].bytes().decode())
    client_leaf = ssl.PEM_cert_to_DER_cert(client_cert.cert_chain_pems[0].bytes().decode())
    issuing_ca = ssl.PEM_cert_to_DER_cert(ca.cert_pem.bytes().decode())
    expected_client_name = x509.load_pem_x509_certificate(
        client_cert.cert_chain_pems[0].bytes()
    ).subject.rfc4514_string()

    assert tls["tls_version"] == TLS_VERSION_1_3
    assert tls["cipher_suite"] in TLS_1_3_CIPHER_SUITES
    assert ssl.PEM_cert_to_DER_cert(tls["server_cert"]) == server_leaf
    assert tls["client_cert_error"] is None
    assert tls["client_cert_name"] == expected_client_name
    # The harvested chain is exactly the certificates the client presented, compared by DER.
    # get_verified_chain (CPython 3.13+) returns the client leaf and the CA; older Pythons
    # fall back to getpeercert and return the leaf alone - matching tests/test_ktls.py.
    chain = [ssl.PEM_cert_to_DER_cert(pem) for pem in tls["client_cert_chain"]]
    assert chain == ([client_leaf, issuing_ca] if sys.version_info >= (3, 13) else [client_leaf])
