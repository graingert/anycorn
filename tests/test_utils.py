"""Tests for anycorn utility functions including header handling and TLS extensions."""

from __future__ import annotations

import socket
import ssl
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
import pytest
import trustme
from anyio import TypedAttributeLookupError
from anyio.abc import SocketStream
from anyio.streams.tls import TLSAttribute, TLSStream

from anycorn.config import Config
from anycorn.utils import (
    ShutdownError,
    build_and_validate_headers,
    build_tls_extension,
    default_tls_extension,
    filter_pseudo_headers,
    is_asgi,
    raise_shutdown,
    suppress_body,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from anycorn.typing import Scope, TLSExtension


@pytest.mark.parametrize(
    ("method", "status", "expected"), [("HEAD", 200, True), ("GET", 200, False), ("GET", 101, True)]
)
def test_suppress_body(method: str, status: int, expected: bool) -> None:  # noqa: FBT001
    assert suppress_body(method, status) is expected


class ASGIClassInstance:
    """ASGI callable class instance for testing is_asgi detection."""

    def __init__(self) -> None:
        pass

    async def __call__(self, scope: Scope, receive: Callable, send: Callable) -> None:
        pass


async def asgi_callable(scope: Scope, receive: Callable, send: Callable) -> None:
    pass


class WSGIClassInstance:
    """WSGI callable class instance for testing is_asgi detection."""

    def __init__(self) -> None:
        pass

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:  # noqa: ARG002
        return []


def wsgi_callable(environ: dict, start_response: Callable) -> Iterable[bytes]:  # noqa: ARG001
    return []


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        (WSGIClassInstance(), False),
        (ASGIClassInstance(), True),
        (wsgi_callable, False),
        (asgi_callable, True),
    ],
)
def test_is_asgi(app: Any, expected: bool) -> None:  # noqa: ANN401, FBT001
    assert is_asgi(app) == expected


def test_build_and_validate_headers_validate() -> None:
    with pytest.raises(TypeError):
        build_and_validate_headers([("string", "string")])  # type: ignore[list-item]


def test_build_and_validate_headers_pseudo() -> None:
    with pytest.raises(ValueError, match="Pseudo headers are not valid"):
        build_and_validate_headers([(b":authority", b"quart")])


def test_filter_pseudo_headers() -> None:
    result = filter_pseudo_headers(
        [(b":authority", b"quart"), (b":path", b"/"), (b"user-agent", b"something")]
    )
    assert result == [(b"host", b"quart"), (b"user-agent", b"something")]


def test_filter_pseudo_headers_no_authority() -> None:
    result = filter_pseudo_headers(
        [(b"host", b"quart"), (b":path", b"/"), (b"user-agent", b"something")]
    )
    assert result == [(b"host", b"quart"), (b"user-agent", b"something")]


class _DummyStream:
    def extra(self, attr: object) -> object:
        raise TypedAttributeLookupError(attr)


def test_build_tls_extension_missing_tls_attributes() -> None:
    config = Config()
    extension = build_tls_extension(config, _DummyStream())  # type: ignore[arg-type]
    assert dict(extension) == default_tls_extension()


class _FakeStream:
    def __init__(self, extras: dict[Any, Any]) -> None:
        self._extras = extras

    def extra(self, attr: Any) -> Any:  # noqa: ANN401
        if attr in self._extras:
            return self._extras[attr]
        raise TypedAttributeLookupError(attr)


class _FakeSSLObject:
    def __init__(
        self,
        der_bytes: bytes | None,
        verify_mode: int = ssl.CERT_OPTIONAL,
        cipher_name: str = "TLS_AES_128_GCM_SHA256",
    ) -> None:
        self._der_bytes = der_bytes
        self._cipher_name = cipher_name
        self.context = SimpleNamespace(verify_mode=verify_mode)

    def get_verified_chain(self) -> tuple[bytes, ...]:
        if self._der_bytes:
            return (self._der_bytes,)
        return ()

    def get_unverified_chain(self) -> tuple[bytes, ...]:
        return self.get_verified_chain()

    def getpeercert(self, binary_form: bool = False) -> Any:  # noqa: ANN401, FBT001, FBT002
        if binary_form:
            return self._der_bytes
        if self._der_bytes:
            return {"subject": ((("commonName", "localhost"),),)}
        return {}

    def cipher(self) -> tuple[str, str, int]:
        return (self._cipher_name, "TLSv1.3", 128)


def test_build_tls_extension_with_client_certificate() -> None:
    pem_cert = Path("tests/assets/cert.pem").read_text()
    der_bytes = ssl.PEM_cert_to_DER_cert(pem_cert)
    fake_ssl = _FakeSSLObject(der_bytes, verify_mode=ssl.CERT_REQUIRED)
    stream = _FakeStream(
        {
            TLSAttribute.tls_version: "TLSv1.3",
            TLSAttribute.ssl_object: fake_ssl,
            TLSAttribute.peer_certificate: {"subject": ((("commonName", "localhost"),),)},
        }
    )
    extension = build_tls_extension(Config(), stream)  # type: ignore[arg-type]
    assert extension["tls_version"] == 0x0304  # noqa: PLR2004
    assert extension["client_cert_chain"]
    assert extension["client_cert_name"] == "CN=localhost"
    assert extension["cipher_suite"] == 0x1301  # noqa: PLR2004
    assert extension["client_cert_error"] is None


def test_build_tls_extension_missing_required_certificate() -> None:
    fake_ssl = _FakeSSLObject(None, verify_mode=ssl.CERT_REQUIRED)
    stream = _FakeStream(
        {
            TLSAttribute.tls_version: "TLSv1.2",
            TLSAttribute.ssl_object: fake_ssl,
        }
    )
    extension = build_tls_extension(Config(), stream)  # type: ignore[arg-type]
    assert extension["client_cert_chain"] == ()
    assert extension["client_cert_name"] is None
    assert extension["client_cert_error"] == "missing-client-certificate"


@pytest.mark.anyio
async def test_build_tls_extension_over_a_real_tls_stream(tmp_path: Path) -> None:  # noqa: PLR0915
    """build_tls_extension harvests the real client certificate off anyio's TLSStream.

    The tests above feed it a hand-written fake ssl object; this drives a real TLS handshake
    over anyio's in-memory TLSStream - the non-kTLS path used for ordinary TLS binds, backed
    by a real ssl.SSLObject - and asserts the extension holds the exact certificates the
    client presented (mirroring tests/test_ktls.py, which does the same over a KTLSStream).
    """
    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost")
    client_cert = ca.issue_cert("client@example.com")
    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    server_cert.cert_chain_pems[0].write_to_path(str(certfile))
    server_cert.private_key_pem.write_to_path(str(keyfile))

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(certfile), str(keyfile))
    server_ctx.verify_mode = ssl.CERT_REQUIRED  # ask the client for a certificate
    ca.configure_trust(server_ctx)
    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ca.configure_trust(client_ctx)
    client_cert.configure_cert(client_ctx)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.setblocking(False)  # noqa: FBT003
    port = listener.getsockname()[1]
    config = Config()
    config.certfile = str(certfile)
    extension: TLSExtension | None = None

    async def run_server() -> None:
        nonlocal extension
        while True:
            try:
                conn, _ = listener.accept()
            except BlockingIOError:  # noqa: PERF203
                await anyio.wait_readable(listener)
            else:
                break
        transport = await SocketStream.from_socket(conn)
        tls = await TLSStream.wrap(
            transport, server_side=True, ssl_context=server_ctx, standard_compatible=False
        )
        extension = build_tls_extension(config, tls)
        await tls.receive()
        await tls.aclose()

    async def run_client() -> None:
        # A blocking stdlib client in a worker thread, so the async server is the code under
        # test and there is no concurrent async close between two anyio streams (racy on trio).
        def talk() -> None:
            with socket.create_connection(("127.0.0.1", port)) as raw:
                tls = client_ctx.wrap_socket(raw, server_hostname="localhost")
                tls.sendall(b"hi")
                tls.close()

        await anyio.to_thread.run_sync(talk)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_server)
        task_group.start_soon(run_client)
    listener.close()

    assert extension is not None
    assert extension["tls_version"] is not None
    assert extension["cipher_suite"] is not None
    assert extension["server_cert"] is not None
    assert extension["client_cert_error"] is None
    assert extension["client_cert_name"] is not None
    assert "O=trustme" in extension["client_cert_name"]
    # The harvested chain is exactly the certificates the client presented, compared by DER.
    # get_verified_chain (CPython 3.13+) returns the client leaf and the CA; older Pythons
    # fall back to getpeercert and return the leaf alone - matching tests/test_ktls.py.
    client_leaf = ssl.PEM_cert_to_DER_cert(client_cert.cert_chain_pems[0].bytes().decode())
    issuing_ca = ssl.PEM_cert_to_DER_cert(ca.cert_pem.bytes().decode())
    chain = [ssl.PEM_cert_to_DER_cert(pem) for pem in extension["client_cert_chain"]]
    assert chain == ([client_leaf, issuing_ca] if sys.version_info >= (3, 13) else [client_leaf])


@pytest.mark.anyio
async def test_raise_shutdown_marks_termination_before_raising() -> None:
    """The marking has to land before the error, or it lands after the servers are gone.

    ShutdownError is what unwinds them, so anything done after it is raised is done
    too late for a server to act on - which is the difference between refusing a
    connection whilst shutting down and accepting one on the way out.
    """
    order: list[str] = []

    async def trigger() -> None:
        order.append("triggered")

    async def on_shutdown() -> None:
        order.append("terminated")

    with pytest.raises(ShutdownError):
        await raise_shutdown(trigger, on_shutdown)

    assert order == ["triggered", "terminated"]


@pytest.mark.anyio
async def test_raise_shutdown_without_a_callback() -> None:
    """The callback stays optional, so existing callers keep working."""

    async def trigger() -> None:
        return

    with pytest.raises(ShutdownError):
        await raise_shutdown(trigger)
