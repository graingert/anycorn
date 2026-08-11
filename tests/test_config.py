"""Tests for anycorn configuration loading and socket creation."""

from __future__ import annotations

import os
import socket
import ssl
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, NonCallableMock

import pytest

import anycorn.config
from anycorn.config import Config, _set_reuse_socket_option

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

    from tests.conftest import TLSCerts

access_log_format = "bob"
h11_max_incomplete_size = 4

# The reuse option _create_sockets sets is platform-specific: SO_EXCLUSIVEADDRUSE on
# Windows (so a second server on a busy port fails), SO_REUSEADDR elsewhere. getattr
# keeps the win32-only constant off the import path on other OSes.
# https://github.com/pgjones/hypercorn/issues/171
_EXPECTED_REUSE_OPTION = (
    getattr(socket, "SO_EXCLUSIVEADDRUSE")  # noqa: B009
    if sys.platform == "win32"
    else socket.SO_REUSEADDR
)


def _check_standard_config(config: Config) -> None:
    assert config.access_log_format == access_log_format
    assert config.h11_max_incomplete_size == h11_max_incomplete_size
    assert config.bind == ["127.0.0.1:5555"]


def test_config_from_pyfile() -> None:
    path = str(Path(__file__).parent / "assets/config.py")
    config = Config.from_pyfile(path)
    _check_standard_config(config)


def test_config_from_object() -> None:
    sys.path.append(str(Path(__file__).parent))

    config = Config.from_object("assets.config")
    _check_standard_config(config)


def test_ssl_config_from_pyfile() -> None:
    path = str(Path(__file__).parent / "assets/config_ssl.py")
    config = Config.from_pyfile(path)
    _check_standard_config(config)
    assert config.ssl_enabled


def test_config_from_toml() -> None:
    path = str(Path(__file__).parent / "assets/config.toml")
    config = Config.from_toml(path)
    _check_standard_config(config)


def test_create_ssl_context() -> None:
    path = str(Path(__file__).parent / "assets/config_ssl.py")
    config = Config.from_pyfile(path)
    context = config.create_ssl_context()

    assert context is not None
    # The hardened options create_default_context() enables, plus OP_NO_COMPRESSION
    # (RFC 7540 9.2.1). Asserting the exact set catches the earlier bug where assigning
    # OP_NO_COMPRESSION cleared everything else, and any future regression that drops one.
    assert context.options == (
        ssl.OP_ALL
        | ssl.OP_CIPHER_SERVER_PREFERENCE
        | ssl.OP_ENABLE_MIDDLEBOX_COMPAT
        | ssl.OP_NO_COMPRESSION
        | ssl.OP_NO_SSLv3
    )


@pytest.mark.parametrize(
    ("bind", "expected_family", "expected_binding"),
    [
        ("127.0.0.1:5000", socket.AF_INET, ("127.0.0.1", 5000)),
        ("127.0.0.1", socket.AF_INET, ("127.0.0.1", 8000)),
        ("[::]:5000", socket.AF_INET6, ("::", 5000)),
        ("[::]", socket.AF_INET6, ("::", 8000)),
    ],
)
def test_create_sockets_ip(
    bind: str,
    expected_family: socket.AddressFamily,
    expected_binding: tuple[str, int],
    monkeypatch: MonkeyPatch,
) -> None:
    mock_socket = Mock()
    monkeypatch.setattr(socket, "socket", mock_socket)
    config = Config()
    config.bind = [bind]
    sockets = config.create_sockets()
    sock = sockets.insecure_sockets[0]
    mock_socket.assert_called_with(expected_family, socket.SOCK_STREAM)
    sock.setsockopt.assert_called_with(socket.SOL_SOCKET, _EXPECTED_REUSE_OPTION, 1)  # type: ignore[attr-defined]
    sock.bind.assert_called_with(expected_binding)  # type: ignore[attr-defined]
    sock.setblocking.assert_called_with(False)  # type: ignore[attr-defined]  # noqa: FBT003
    sock.set_inheritable.assert_called_with(True)  # type: ignore[attr-defined]  # noqa: FBT003


@pytest.mark.skipif(sys.platform == "win32", reason="Windows is not Unix.")
def test_create_sockets_unix(monkeypatch: MonkeyPatch) -> None:
    mock_socket = Mock()
    monkeypatch.setattr(socket, "socket", mock_socket)
    monkeypatch.setattr(os, "chown", Mock())
    config = Config()
    config.bind = ["unix:/tmp/anycorn.sock"]
    sockets = config.create_sockets()
    sock = sockets.insecure_sockets[0]
    mock_socket.assert_called_with(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.setsockopt.assert_called_with(socket.SOL_SOCKET, _EXPECTED_REUSE_OPTION, 1)  # type: ignore[attr-defined]
    sock.bind.assert_called_with("/tmp/anycorn.sock")  # type: ignore[attr-defined] # noqa: S108
    sock.setblocking.assert_called_with(False)  # type: ignore[attr-defined]  # noqa: FBT003
    sock.set_inheritable.assert_called_with(True)  # type: ignore[attr-defined]  # noqa: FBT003


@pytest.mark.skipif(sys.platform == "win32", reason="Windows is not Unix.")
def test_create_sockets_unix_restores_umask_on_bind_failure(monkeypatch: MonkeyPatch) -> None:
    """A failed bind must not leak the configured umask into the rest of the process."""
    mock_socket = Mock()
    mock_socket.return_value.bind.side_effect = OSError("bind failed")
    monkeypatch.setattr(socket, "socket", mock_socket)

    umask_calls: list[int] = []

    def fake_umask(mask: int) -> int:
        umask_calls.append(mask)
        return 0o22

    monkeypatch.setattr(os, "umask", fake_umask)

    config = Config()
    config.umask = 0o077
    config.bind = ["unix:/tmp/anycorn.sock"]

    with pytest.raises(OSError, match="bind failed"):
        config.create_sockets()

    # The configured umask is applied, then restored to what os.umask returned even
    # though bind() raised in between.
    assert umask_calls == [0o077, 0o22]


def test_create_sockets_fd(monkeypatch: MonkeyPatch) -> None:
    mock_sock_class = Mock(
        return_value=NonCallableMock(**{"getsockopt.return_value": socket.SOCK_STREAM})  # type: ignore[arg-type]
    )
    monkeypatch.setattr(socket, "socket", mock_sock_class)
    config = Config()
    config.bind = ["fd://2"]
    sockets = config.create_sockets()
    sock = sockets.insecure_sockets[0]
    mock_sock_class.assert_called_with(fileno=2)
    sock.getsockopt.assert_called_with(socket.SOL_SOCKET, socket.SO_TYPE)  # type: ignore[attr-defined]
    sock.setsockopt.assert_called_with(socket.SOL_SOCKET, _EXPECTED_REUSE_OPTION, 1)  # type: ignore[attr-defined]
    sock.setblocking.assert_called_with(False)  # type: ignore[attr-defined]  # noqa: FBT003
    sock.set_inheritable.assert_called_with(True)  # type: ignore[attr-defined]  # noqa: FBT003


@pytest.mark.skipif(sys.platform == "win32", reason="Windows is not Unix.")
def test_create_sockets_multiple(monkeypatch: MonkeyPatch) -> None:
    mock_socket = Mock()
    monkeypatch.setattr(socket, "socket", mock_socket)
    monkeypatch.setattr(os, "chown", Mock())
    config = Config()
    config.bind = ["127.0.0.1", "unix:/tmp/anycorn.sock"]
    sockets = config.create_sockets()
    assert len(sockets.insecure_sockets) == 2  # noqa: PLR2004


@pytest.mark.skipif(sys.platform == "win32", reason="SO_REUSEADDR is the Unix path.")
def test_set_reuse_socket_option_posix_sets_reuseaddr() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _set_reuse_socket_option(sock)
        # getsockopt only guarantees a nonzero value for an enabled boolean option;
        # macOS/BSD returns something other than 1, so assert truthiness, not == 1.
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0
    finally:
        sock.close()


def test_set_reuse_socket_option_windows_uses_exclusive_addr_use(
    monkeypatch: MonkeyPatch,
) -> None:
    """On Windows the port is claimed exclusively, never with SO_REUSEADDR.

    SO_EXCLUSIVEADDRUSE only exists on Windows, so it is patched in here to exercise
    the branch on any platform; the point is that SO_REUSEADDR - which lets a second
    server hijack the port on Windows - is not the option that gets set.

    https://github.com/pgjones/hypercorn/issues/171
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", -5, raising=False)
    sock = Mock()

    _set_reuse_socket_option(sock)

    exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE")  # noqa: B009  # win32-only constant
    sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, exclusive, 1)


def test_second_server_on_the_same_port_is_refused(free_tcp_port: int) -> None:
    """A second server must not be able to hijack a port already being served.

    Binds a real server socket and starts it listening, then a second socket claims the
    same address exactly as _create_sockets does (via _set_reuse_socket_option) and must
    fail to bind. On Unix SO_REUSEADDR already refuses this, so it passes regardless; the
    fix is what makes Windows behave the same way with SO_EXCLUSIVEADDRUSE rather than
    letting the second bind steal the port - so without the fix this fails on Windows.

    https://github.com/pgjones/hypercorn/issues/171
    """
    first = Config()
    first.bind = [f"127.0.0.1:{free_tcp_port}"]
    sockets = first.create_sockets()
    try:
        for listening in sockets.insecure_sockets:
            listening.listen(first.backlog)

        second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            _set_reuse_socket_option(second)
            with pytest.raises(OSError):  # noqa: PT011  # EADDRINUSE / WSAEADDRINUSE
                second.bind(("127.0.0.1", free_tcp_port))
        finally:
            second.close()
    finally:
        for sock in sockets.insecure_sockets:
            sock.close()


def test_daemon_defaults_true() -> None:
    assert Config().daemon is True


def test_response_headers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(anycorn.config, "time", lambda: 1_512_229_395)
    config = Config()
    assert config.response_headers("test") == [
        (b"date", b"Sat, 02 Dec 2017 15:43:15 GMT"),
        (b"server", b"anycorn-test"),
    ]
    config.include_server_header = False
    assert config.response_headers("test") == [(b"date", b"Sat, 02 Dec 2017 15:43:15 GMT")]


def test_response_headers_without_the_date_header() -> None:
    """With include_date_header off, only the server header remains."""
    config = Config()
    config.include_date_header = False
    assert config.response_headers("test") == [(b"server", b"anycorn-test")]


def test_config_from_object_dotted_instance_path() -> None:
    """A "module.attr" string imports the module and takes the named attribute off it."""
    sys.path.append(str(Path(__file__).parent))
    config = Config.from_object("assets.config.instance")
    _check_standard_config(config)


def test_from_mapping_accepts_keyword_arguments_alone() -> None:
    """With no mapping given, the keyword arguments become the mapping."""
    config = Config.from_mapping(keep_alive_timeout=10)
    assert config.keep_alive_timeout == 10  # noqa: PLR2004


def test_cert_reqs_setter_is_deprecated_and_sets_verify_mode() -> None:
    """The legacy cert_reqs property warns and writes through to verify_mode."""
    config = Config()
    with pytest.warns(Warning, match="Please use verify_mode instead"):
        config.cert_reqs = int(ssl.CERT_REQUIRED)
    assert config.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("attr", ["insecure_bind", "quic_bind"])
def test_bind_setters_wrap_a_bare_string_in_a_list(attr: str) -> None:
    """A single string is stored as a one-element list; a list is stored as-is."""
    config = Config()
    setattr(config, attr, "127.0.0.1:1234")
    assert getattr(config, attr) == ["127.0.0.1:1234"]
    setattr(config, attr, ["127.0.0.1:1", "127.0.0.1:2"])
    assert getattr(config, attr) == ["127.0.0.1:1", "127.0.0.1:2"]


def test_create_ssl_context_is_none_without_a_certificate() -> None:
    """No certfile/keyfile means TLS is off, so there is no context to build."""
    assert Config().create_ssl_context() is None


def test_create_ssl_context_applies_verification_settings(tls_certs: TLSCerts) -> None:
    """ca_certs, verify_mode and verify_flags are all threaded onto the context."""
    config = Config()
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    config.ca_certs = str(tls_certs.cafile)
    config.verify_mode = ssl.CERT_REQUIRED
    config.verify_flags = ssl.VERIFY_X509_STRICT

    context = config.create_ssl_context()

    assert context is not None
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.verify_flags & ssl.VERIFY_X509_STRICT


def test_create_sockets_reuseport_for_multiple_workers(monkeypatch: MonkeyPatch) -> None:
    """With more than one worker the listening socket is opened with SO_REUSEPORT."""
    mock_socket = Mock()
    monkeypatch.setattr(socket, "socket", mock_socket)
    config = Config()
    config.workers = 2
    config.bind = ["127.0.0.1:5000"]
    config.create_sockets()
    sock = mock_socket.return_value
    sock.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)


def test_create_sockets_fd_wrong_type_raises(monkeypatch: MonkeyPatch) -> None:
    """A file descriptor of the wrong socket kind is rejected with SocketTypeError."""
    mock_sock_class = Mock(
        return_value=NonCallableMock(**{"getsockopt.return_value": socket.SOCK_DGRAM})  # type: ignore[arg-type]
    )
    monkeypatch.setattr(socket, "socket", mock_sock_class)
    config = Config()
    config.bind = ["fd://2"]  # a datagram fd offered where a stream socket is wanted
    with pytest.raises(anycorn.config.SocketTypeError, match="Unexpected socket type"):
        config.create_sockets()


def test_set_quic_addresses_warns_on_an_unusable_socket_name() -> None:
    """A socket whose name is not a host/port pair cannot yield an alt-svc header."""
    config = Config()
    sock = Mock()
    sock.getsockname.return_value = "a-unix-path"  # not a (host, port) tuple
    with pytest.warns(Warning, match="Cannot create a alt-svc header"):
        config._set_quic_addresses([sock])
    assert config._quic_addresses == []


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX is a Unix concept.")
def test_create_sockets_unix_unlinks_a_stale_socket_file(tmp_path: Path) -> None:
    """A leftover socket file at the bind path is removed before rebinding."""
    sock_path = tmp_path / "stale.sock"
    # Leave a real socket file behind, exactly as a crashed previous server would.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sock_path))
    stale.close()
    assert sock_path.exists()

    config = Config()
    config.bind = [f"unix:{sock_path}"]
    sockets = config.create_sockets()
    try:
        # Rebinding only succeeds because the stale file was unlinked first.
        assert len(sockets.insecure_sockets) == 1
        assert stat.S_ISSOCK(sock_path.stat().st_mode)
    finally:
        for sock in sockets.insecure_sockets:
            sock.close()


@pytest.mark.skipif(sys.platform == "win32", reason="os.chown is a Unix concept.")
def test_create_sockets_unix_chowns_the_socket(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """When user and group are set the bound socket file is chowned to them."""
    sock_path = tmp_path / "owned.sock"
    config = Config()
    config.bind = [f"unix:{sock_path}"]
    # Chown to our own ids, the one target an unprivileged test may use.
    config.user = os.getuid()
    config.group = os.getgid()

    chown_calls: list[tuple[str, int, int]] = []
    real_chown = os.chown

    def recording_chown(path: str, uid: int, gid: int) -> None:
        chown_calls.append((str(path), uid, gid))
        real_chown(path, uid, gid)

    monkeypatch.setattr(os, "chown", recording_chown)
    sockets = config.create_sockets()
    try:
        assert chown_calls == [(str(sock_path), os.getuid(), os.getgid())]
    finally:
        for sock in sockets.insecure_sockets:
            sock.close()
