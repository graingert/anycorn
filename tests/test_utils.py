"""Tests for anycorn utility functions including header handling and TLS extensions."""

from __future__ import annotations

import os
import socket
import ssl
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import trustme
from anyio import TypedAttributeLookupError
from anyio.streams.tls import TLSAttribute

from anycorn.app_wrappers import ASGIWrapper, WSGIWrapper
from anycorn.config import Config
from anycorn.utils import (
    NoAppError,
    ShutdownError,
    _cached_server_cert,
    _escape_rfc4514_value,
    _extract_client_chain,
    _subject_to_rfc4514,
    build_and_validate_headers,
    build_tls_extension,
    check_for_updates,
    check_multiprocess_shutdown_event,
    default_tls_extension,
    files_to_watch,
    filter_pseudo_headers,
    is_asgi,
    load_application,
    parse_socket_addr,
    raise_shutdown,
    repr_socket_addr,
    suppress_body,
    tls_version_to_int,
    valid_server_name,
    write_pid_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from anycorn.typing import Scope


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


@pytest.mark.anyio
async def test_is_asgi_doubles_are_callable() -> None:
    """Exercise the test doubles themselves so their bodies are covered."""
    # The doubles ignore their arguments, so the callbacks are never invoked.
    await asgi_callable({}, None, None)  # type: ignore[arg-type]
    await ASGIClassInstance()({}, None, None)  # type: ignore[arg-type]
    assert wsgi_callable({}, None) == []  # type: ignore[arg-type]
    assert WSGIClassInstance()({}, None) == []  # type: ignore[arg-type]


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
        # Only the binary form is consulted (via _extract_client_chain); the fake is not a
        # real ssl.SSLObject, so build_tls_extension never calls the dict form on it.
        return self._der_bytes if binary_form else None

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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0x0304, 0x0304),  # an int passes straight through
        ("TLSv1.3", 0x0304),  # a named version from the map
        ("TLSv1", 0x0301),
        ("TLSv1.4", 0x0305),  # a TLSv1.x not in the map is computed
        ("TLSv1.0", 0x0301),
        ("TLSvX", None),  # a TLSv prefix with a non-numeric minor
        ("nonsense", None),
        (None, None),
    ],
)
def test_tls_version_to_int(value: str | int | None, expected: int | None) -> None:
    assert tls_version_to_int(value) == expected


def test_escape_rfc4514_value_escapes_special_and_edge_characters() -> None:
    assert _escape_rfc4514_value("a,b+c") == "a\\,b\\+c"
    assert _escape_rfc4514_value("#lead") == "\\#lead"  # leading '#'
    assert _escape_rfc4514_value(" lead") == "\\ lead"  # leading space
    assert _escape_rfc4514_value("trail ") == "trail\\ "  # trailing space
    assert _escape_rfc4514_value("plain") == "plain"


def test_subject_to_rfc4514_returns_none_on_malformed_input() -> None:
    assert _subject_to_rfc4514([(("commonName", "localhost"),)]) == "CN=localhost"
    # A subject whose iteration blows up is reported as no name rather than raising.
    assert _subject_to_rfc4514(object()) is None  # type: ignore[arg-type]
    assert _subject_to_rfc4514([]) is None  # nothing to render


@pytest.mark.parametrize("app", [42, "not-callable", object()])
def test_is_asgi_is_false_for_non_callables(app: Any) -> None:  # noqa: ANN401
    assert is_asgi(app) is False


# AF_UNIX does not exist on Windows, so guard the reference and skip those cases there.
_AF_UNIX = getattr(socket, "AF_UNIX", None)
_needs_af_unix = pytest.mark.skipif(_AF_UNIX is None, reason="AF_UNIX is unavailable")


@pytest.mark.parametrize(
    ("family", "address", "expected"),
    [
        (socket.AF_INET, ("127.0.0.1", 80), ("127.0.0.1", 80)),
        (socket.AF_INET6, ("::1", 80, 0, 0), ("::1", 80)),
        pytest.param(_AF_UNIX, "anycorn.sock", None, marks=_needs_af_unix),
    ],
)
def test_parse_socket_addr(family: int, address: Any, expected: Any) -> None:  # noqa: ANN401
    assert parse_socket_addr(family, address) == expected


@pytest.mark.parametrize(
    ("family", "address", "expected"),
    [
        (socket.AF_INET, ("127.0.0.1", 80), "127.0.0.1:80"),
        (socket.AF_INET6, ("::1", 80), "[::1]:80"),
        pytest.param(_AF_UNIX, "anycorn.sock", "unix:anycorn.sock", marks=_needs_af_unix),
        (-1, ("weird",), "('weird',)"),  # an unknown family falls back to repr
    ],
)
def test_repr_socket_addr(family: int, address: Any, expected: str) -> None:  # noqa: ANN401
    assert repr_socket_addr(family, address) == expected


def test_valid_server_name() -> None:
    config = Config()
    request = SimpleNamespace(headers=[(b"x-other", b"1"), (b"host", b"allowed")])

    config.server_names = []
    assert valid_server_name(config, request) is True  # type: ignore[arg-type]

    config.server_names = ["allowed"]
    assert valid_server_name(config, request) is True  # type: ignore[arg-type]

    config.server_names = ["elsewhere"]
    assert valid_server_name(config, request) is False  # type: ignore[arg-type]

    # No host header at all leaves the host empty, which matches no configured name.
    request_no_host = SimpleNamespace(headers=[(b"x-other", b"1")])
    assert valid_server_name(config, request_no_host) is False  # type: ignore[arg-type]


def test_write_pid_file(tmp_path: Path) -> None:
    pid_path = tmp_path / "anycorn.pid"
    write_pid_file(str(pid_path))
    assert pid_path.read_text() == str(os.getpid())


def test_files_to_watch_maps_loaded_modules_to_mtimes() -> None:
    watched = files_to_watch()
    # anycorn.utils is imported, has a real file, and so must be watched.
    assert any(path.name == "utils.py" for path in watched)
    assert all(isinstance(mtime, float) for mtime in watched.values())


def test_check_for_updates(tmp_path: Path) -> None:
    watched_file = tmp_path / "watched.py"
    watched_file.write_text("x = 1")
    path = Path(watched_file)

    # A file whose mtime advanced is reported as changed.
    assert check_for_updates({path: path.stat().st_mtime - 10}) is True

    # With the recorded mtime current, nothing has changed.
    assert check_for_updates({path: path.stat().st_mtime}) is False

    # A file that has vanished counts as a change (so the reloader restarts).
    assert check_for_updates({tmp_path / "gone.py": 0.0}) is True


@pytest.mark.anyio
async def test_check_multiprocess_shutdown_event_polls_until_set() -> None:
    slept: list[float] = []

    class _Event:
        def __init__(self) -> None:
            self._checks = 0

        def is_set(self) -> bool:
            self._checks += 1
            return self._checks > 1  # unset on the first poll, set on the second

    async def sleep(delay: float) -> None:
        slept.append(delay)

    await check_multiprocess_shutdown_event(_Event(), sleep)  # type: ignore[arg-type]
    assert slept == [0.1]  # it slept once, between the two polls


def test_load_application_infers_asgi_and_wsgi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app_infer_asgi.py").write_text("async def app(scope, receive, send):\n    pass\n")
    (tmp_path / "app_infer_wsgi.py").write_text(
        "def app(environ, start_response):\n    return []\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(load_application("app_infer_asgi", 0), ASGIWrapper)
    assert isinstance(load_application("app_infer_wsgi", 0), WSGIWrapper)


def test_load_application_honours_an_explicit_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app_explicit.py").write_text("def app(environ, start_response):\n    return []\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    # An async app forced to wsgi is wrapped as wsgi; the mode wins over inference.
    assert isinstance(load_application("wsgi:app_explicit:app", 0), WSGIWrapper)


def test_load_application_rejects_an_unknown_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app_mode.py").write_text("app = None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="Invalid mode"):
        load_application("bogus:app_mode:app", 0)


def test_load_application_reports_a_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.path", ["/nonexistent", *__import__("sys").path])
    with pytest.raises(NoAppError, match="module not found"):
        load_application("no_such_anycorn_module_xyz", 0)


def test_load_application_reports_a_missing_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app_missing_attr.py").write_text("app = None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(NoAppError, match="application not found"):
        load_application("app_missing_attr:not_here", 0)


def test_load_application_propagates_a_nested_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dependency inside the app module is not mistaken for a missing app module."""
    (tmp_path / "app_nested.py").write_text("import definitely_not_a_real_module_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError):
        load_application("app_nested", 0)


def test_cached_server_cert_returns_none_for_unreadable_or_certless_files(tmp_path: Path) -> None:
    assert _cached_server_cert(str(tmp_path / "missing.pem")) is None  # OSError path
    certless = tmp_path / "certless.pem"
    certless.write_text("no certificate here")
    assert _cached_server_cert(str(certless)) is None  # no BEGIN CERTIFICATE block


def test_build_tls_extension_includes_the_server_certificate() -> None:
    config = Config()
    config.certfile = "tests/assets/cert.pem"
    extension = build_tls_extension(config, _DummyStream())  # type: ignore[arg-type]
    assert extension["server_cert"] is not None
    assert "BEGIN CERTIFICATE" in extension["server_cert"]


class _RaisingSSL:
    """An ssl-object stand-in whose every accessor raises, exercising the guard clauses."""

    def get_verified_chain(self) -> tuple[bytes, ...]:
        raise ssl.SSLError

    def get_unverified_chain(self) -> tuple[bytes, ...]:
        raise ssl.SSLError

    def getpeercert(self, binary_form: bool = False) -> Any:  # noqa: ANN401, FBT001, FBT002, ARG002
        raise ssl.SSLError

    def cipher(self) -> tuple[str, str, int]:
        raise ssl.SSLError

    # deliberately no ``context`` attribute, to drive the AttributeError guard


def test_extract_client_chain_survives_raising_chain_methods() -> None:
    assert _extract_client_chain(_RaisingSSL()) == ()  # type: ignore[arg-type]


class _PeerCertOnlySSL:
    """No chain methods; the peer cert is only reachable through getpeercert(binary_form)."""

    def __init__(self, der: bytes) -> None:
        self._der = der

    def getpeercert(self, binary_form: bool = False) -> Any:  # noqa: ANN401, FBT001, FBT002
        return self._der if binary_form else {}


def test_extract_client_chain_falls_back_to_getpeercert() -> None:
    der = ssl.PEM_cert_to_DER_cert(Path("tests/assets/cert.pem").read_text())
    chain = _extract_client_chain(_PeerCertOnlySSL(der))  # type: ignore[arg-type]
    assert len(chain) == 1
    assert "BEGIN CERTIFICATE" in chain[0]


def test_build_tls_extension_guards_against_raising_ssl_accessors() -> None:
    """A cipher() that raises and a missing context must not blow up extension building."""
    stream = _FakeStream({TLSAttribute.ssl_object: _RaisingSSL()})
    extension = build_tls_extension(Config(), stream)  # type: ignore[arg-type]
    assert extension["cipher_suite"] is None
    assert extension["client_cert_chain"] == ()


class _ChainWithoutNameSSL:
    """Presents a chain but no subject, so a name cannot be derived from it."""

    def __init__(self, der: bytes) -> None:
        self._der = der

    def get_verified_chain(self) -> tuple[bytes, ...]:
        return (self._der,)  # the chain is found here, so getpeercert is never consulted

    def cipher(self) -> tuple[str, str, int]:
        return ("TLS_AES_128_GCM_SHA256", "TLSv1.3", 128)

    context = SimpleNamespace(verify_mode=ssl.CERT_OPTIONAL)


def test_build_tls_extension_leaves_name_none_when_chain_has_no_subject() -> None:
    der = ssl.PEM_cert_to_DER_cert(Path("tests/assets/cert.pem").read_text())
    stream = _FakeStream({TLSAttribute.ssl_object: _ChainWithoutNameSSL(der)})
    extension = build_tls_extension(Config(), stream)  # type: ignore[arg-type]
    assert extension["client_cert_chain"]  # a chain was harvested
    assert extension["client_cert_name"] is None  # but no name could be derived


def _mtls_server_ssl_object(*, client_presents_cert: bool = True) -> ssl.SSLObject:
    """Complete a real in-memory TLS handshake and return the server SSLObject.

    With ``client_presents_cert`` the client authenticates and the server can read its
    subject; without it the handshake still completes (the server only requests a cert),
    exercising the path where ``getpeercert()`` yields no subject.
    """
    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost")
    # A common_name puts the identity in the subject, where getpeercert() reads it.
    client_cert = ca.issue_cert("client@example.com", common_name="client.example.com")

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_cert.configure_cert(server_ctx)
    server_ctx.verify_mode = ssl.CERT_REQUIRED if client_presents_cert else ssl.CERT_OPTIONAL
    ca.configure_trust(server_ctx)

    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ca.configure_trust(client_ctx)
    if client_presents_cert:
        client_cert.configure_cert(client_ctx)

    c_in, c_out, s_in, s_out = (ssl.MemoryBIO() for _ in range(4))
    client = client_ctx.wrap_bio(c_in, c_out, server_hostname="localhost")
    server = server_ctx.wrap_bio(s_in, s_out, server_side=True)

    done = {client: False, server: False}
    while not all(done.values()):
        for obj, out_bio, peer_in in ((client, c_out, s_in), (server, s_out, c_in)):
            try:
                obj.do_handshake()
            except ssl.SSLWantReadError:
                pass
            else:
                done[obj] = True
            peer_in.write(out_bio.read())
    return server


def test_build_tls_extension_derives_name_from_a_real_getpeercert() -> None:
    """With a real SSLObject and no peer_certificate attribute, the name comes from getpeercert."""
    server = _mtls_server_ssl_object()
    stream = _FakeStream({TLSAttribute.ssl_object: server})  # no peer_certificate attr
    extension = build_tls_extension(Config(), stream)  # type: ignore[arg-type]
    name = extension["client_cert_name"]
    assert name is not None
    assert name.startswith("CN=client.example.com,")  # subject DN harvested via getpeercert


def test_build_tls_extension_leaves_name_none_without_a_client_certificate() -> None:
    """A real handshake where the client sends no cert yields an empty getpeercert subject."""
    server = _mtls_server_ssl_object(client_presents_cert=False)
    stream = _FakeStream({TLSAttribute.ssl_object: server})
    extension = build_tls_extension(Config(), stream)  # type: ignore[arg-type]
    assert extension["client_cert_name"] is None
    assert extension["client_cert_chain"] == ()
