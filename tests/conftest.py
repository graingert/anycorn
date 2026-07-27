"""Shared pytest fixtures for Anycorn tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest
import trustme

import anycorn.config
from anycorn.protocol.httptools_connection import httptools_available
from anycorn.typing import ConnectionState, HTTPScope

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


class TLSCerts(NamedTuple):
    """Paths to a freshly issued server certificate and the CA that signed it."""

    certfile: Path
    keyfile: Path
    cafile: Path
    hostname: str


@pytest.fixture(autouse=True)
def _time(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(anycorn.config, "time", lambda: 5000)


@pytest.fixture(autouse=True, params=["h11", "httptools"])
def http_parser(request: pytest.FixtureRequest, monkeypatch: MonkeyPatch) -> str:
    """Run the whole suite against both HTTP/1.1 parsers.

    The point of the httptools support is that nothing above the connection can
    tell which one is in use, so the way to keep that true is to assert it of
    every test rather than of a chosen few.
    """
    if request.param == "httptools" and not httptools_available():
        pytest.skip("httptools is not installed")
    monkeypatch.setattr(anycorn.config.Config, "http_parser", request.param)
    return request.param


@pytest.fixture(name="tls_certs", scope="session")
def _tls_certs(tmp_path_factory: pytest.TempPathFactory) -> TLSCerts:
    """Issue a server certificate for the tests that actually negotiate TLS.

    Generated per run rather than checked in, so there is nothing to expire, and so
    the clients can verify the chain properly instead of disabling verification.
    """
    ca = trustme.CA()
    hostname = "localhost"
    server_cert = ca.issue_cert(hostname, "127.0.0.1")

    path = tmp_path_factory.mktemp("tls")
    certs = TLSCerts(path / "cert.pem", path / "key.pem", path / "ca.pem", hostname)
    server_cert.cert_chain_pems[0].write_to_path(certs.certfile)
    server_cert.private_key_pem.write_to_path(certs.keyfile)
    ca.cert_pem.write_to_path(certs.cafile)
    return certs


@pytest.fixture(name="http_scope")
def _http_scope() -> HTTPScope:
    return {
        "type": "http",
        "asgi": {},
        "http_version": "2",
        "method": "GET",
        "scheme": "https",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"a=b",
        "root_path": "",
        "headers": [
            (b"User-Agent", b"Anycorn"),
            (b"X-Anycorn", b"Anycorn"),
            (b"Referer", b"anycorn"),
        ],
        "client": ("127.0.0.1", 80),
        "server": None,
        "extensions": {},
        "state": ConnectionState({}),
    }
