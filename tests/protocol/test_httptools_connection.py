"""Tests for the httptools-backed connection.

The contract is not "behaves reasonably" but "behaves exactly as the h11-backed
one does", since `H11Protocol` and everything above it reads the events and the
bytes without knowing which parser produced them. So most of this drives the two
connections side by side rather than asserting literals, which would only record
whatever this implementation happened to do.
"""

from __future__ import annotations

import pytest

from anycorn.config import Config
from anycorn.protocol import http1_events as http1
from anycorn.protocol.h11 import _make_connection
from anycorn.protocol.h11_connection import H11Connection
from anycorn.protocol.httptools_connection import HttpToolsConnection

GET = b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"
GET_10 = b"GET / HTTP/1.0\r\nHost: anycorn\r\n\r\n"
HEAD = b"HEAD / HTTP/1.1\r\nHost: anycorn\r\n\r\n"
POST = b"POST / HTTP/1.1\r\nHost: anycorn\r\ncontent-length: 4\r\n\r\nbody"


def _drive(connection: object, request: bytes | None, sends: list) -> tuple[list, bytes]:
    """Feed *request* in, take the events out, then encode *sends* and keep the bytes."""
    events = []
    if request is not None:
        connection.receive_data(request)  # type: ignore[attr-defined]
        while True:
            event = connection.next_event()  # type: ignore[attr-defined]
            events.append(event)
            if event in (http1.NEED_DATA, http1.PAUSED) or isinstance(
                event, (http1.EndOfMessage, http1.ConnectionClosed)
            ):
                break

    return events, b"".join(_encode(connection, event) for event in sends)


def _next_request(connection: H11Connection | HttpToolsConnection) -> http1.Request:
    """Take the next event, which the caller expects to be a request."""
    event = connection.next_event()
    assert isinstance(event, http1.Request), f"expected a Request, got {event!r}"
    return event


def _encode(connection: object, event: object) -> bytes:
    try:
        return connection.send(event)  # type: ignore[attr-defined]
    except http1.LocalProtocolError:
        return b"<LocalProtocolError>"


@pytest.mark.parametrize(
    ("request_bytes", "sends"),
    [
        # A response before any request has arrived, as an error response is
        (None, [http1.Response(status_code=201, headers=[]), http1.EndOfMessage()]),
        (
            GET,
            [
                http1.Response(status_code=200, headers=[(b"content-length", b"4")]),
                http1.Data(data=b"body"),
                http1.EndOfMessage(),
            ],
        ),
        # No framing given, so the connection has to choose one
        (
            GET,
            [
                http1.Response(status_code=200, headers=[]),
                http1.Data(data=b"body"),
                http1.EndOfMessage(),
            ],
        ),
        # HTTP/1.0 cannot be sent chunked, so it is delimited by closing
        (
            GET_10,
            [
                http1.Response(status_code=200, headers=[]),
                http1.Data(data=b"body"),
                http1.EndOfMessage(),
            ],
        ),
        # Framed as the equivalent GET, but no body written
        (HEAD, [http1.Response(status_code=200, headers=[]), http1.EndOfMessage()]),
        (GET, [http1.Response(status_code=204, headers=[]), http1.EndOfMessage()]),
        (GET, [http1.Response(status_code=304, headers=[]), http1.EndOfMessage()]),
        # The case the app wrote its header names in is preserved
        (
            GET,
            [
                http1.Response(status_code=201, headers=[(b"X-Special", b"Value")]),
                http1.EndOfMessage(),
            ],
        ),
        # An app that closes the connection itself must not get two of them
        (
            GET,
            [
                http1.Response(
                    status_code=200, headers=[(b"content-length", b"0"), (b"connection", b"close")]
                ),
                http1.EndOfMessage(),
            ],
        ),
        (
            POST,
            [
                http1.Response(status_code=200, headers=[(b"content-length", b"2")]),
                http1.Data(data=b"ok"),
                http1.EndOfMessage(),
            ],
        ),
        (
            b"POST / HTTP/1.1\r\nHost: a\r\nExpect: 100-continue\r\ncontent-length: 4\r\n\r\n",
            [http1.InformationalResponse(status_code=100, headers=[])],
        ),
    ],
)
def test_writes_the_same_bytes_as_h11(request_bytes: bytes | None, sends: list) -> None:
    """Byte for byte, so nothing downstream can tell which parser answered."""
    _, mine = _drive(HttpToolsConnection(16 * 1024), request_bytes, sends)
    _, theirs = _drive(H11Connection(16 * 1024), request_bytes, sends)

    assert mine == theirs


@pytest.mark.parametrize("request_bytes", [GET, GET_10, HEAD, POST])
def test_reads_the_same_events_as_h11(request_bytes: bytes) -> None:
    """The event stream has to match too, since H11Protocol switches on its types."""
    mine, _ = _drive(HttpToolsConnection(16 * 1024), request_bytes, [])
    theirs, _ = _drive(H11Connection(16 * 1024), request_bytes, [])

    assert [type(event).__name__ for event in mine] == [type(event).__name__ for event in theirs]
    first_mine, first_theirs = mine[0], theirs[0]
    assert isinstance(first_mine, http1.Request)
    assert isinstance(first_theirs, http1.Request)
    assert first_mine.method == first_theirs.method
    assert first_mine.target == first_theirs.target
    assert first_mine.http_version == first_theirs.http_version
    assert list(first_mine.headers) == list(first_theirs.headers)


def test_a_second_pipelined_request_pauses_rather_than_merging() -> None:
    """Httptools reads straight on into the next request; h11 stops and pauses.

    Left alone the two get folded together, which surfaces as a duplicate-header
    error rather than as pipelining - and the second request is lost.
    """
    keep_alive = b"GET /first HTTP/1.1\r\nHost: a\r\nConnection: keep-alive\r\n\r\n"
    connection = HttpToolsConnection(16 * 1024)
    connection.receive_data(keep_alive + b"GET /second HTTP/1.1\r\nHost: a\r\n\r\n")

    assert _next_request(connection).target == b"/first"
    assert isinstance(connection.next_event(), http1.EndOfMessage)
    assert connection.next_event() is http1.PAUSED

    # Recycling needs both sides finished, which is what H11Protocol waits for
    connection.send(http1.Response(status_code=200, headers=[(b"content-length", b"0")]))
    connection.send(http1.EndOfMessage())
    connection.start_next_cycle()

    assert _next_request(connection).target == b"/second"


def test_data_after_a_closing_request_is_an_error() -> None:
    """Which is what h11 answers, rather than treating it as the next request."""
    connection = HttpToolsConnection(16 * 1024)
    connection.receive_data(b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\r\n")
    assert _next_request(connection).target == b"/"
    assert isinstance(connection.next_event(), http1.EndOfMessage)

    connection.receive_data(b"some body")

    with pytest.raises(http1.RemoteProtocolError):
        connection.next_event()


def test_an_oversized_header_block_is_rejected() -> None:
    """The limit h11 enforces via max_incomplete_event_size, with h11's status."""
    connection = HttpToolsConnection(64)
    connection.receive_data(b"GET / HTTP/1.1\r\nHost: a\r\nX-Long: " + b"a" * 200)

    with pytest.raises(http1.RemoteProtocolError) as exc_info:
        connection.next_event()

    assert exc_info.value.error_status_hint == 431  # noqa: PLR2004


def test_the_http2_preface_is_handed_over_as_a_request() -> None:
    """Httptools rejects it outright, where h11 parses it and lets the caller decide.

    H11Protocol looks for exactly this to spot a client speaking HTTP/2 with prior
    knowledge, so the preface has to arrive as a request rather than as a 400.
    """
    connection = HttpToolsConnection(16 * 1024)
    connection.receive_data(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")

    event = _next_request(connection)

    assert event.method == b"PRI"
    assert event.target == b"*"
    assert event.http_version == b"2.0"
    assert connection.trailing_data[0] == b"SM\r\n\r\n"


@pytest.mark.parametrize(
    ("parser", "expected"),
    [("h11", H11Connection), ("httptools", HttpToolsConnection), ("auto", H11Connection)],
)
def test_the_config_chooses_the_parser(parser: str, expected: type) -> None:
    config = Config()
    config.http_parser = parser  # type: ignore[assignment]

    assert isinstance(_make_connection(config), expected)


def test_asking_for_httptools_without_it_installed_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rather than quietly serving with h11, which is the opposite of what was asked."""
    monkeypatch.setattr("anycorn.protocol.h11.httptools_available", lambda: False)
    config = Config()
    config.http_parser = "httptools"

    with pytest.raises(RuntimeError, match="httptools is not installed"):
        _make_connection(config)


def test_auto_uses_h11_even_with_httptools_installed() -> None:
    """Installing httptools must not quietly change how requests are parsed.

    h11 is present either way, since wsproto requires it, so preferring it means
    adding the extra is a decision rather than a side effect.
    """
    config = Config()
    config.http_parser = "auto"

    assert isinstance(_make_connection(config), H11Connection)


def test_asking_for_h11_without_it_installed_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mirror of the httptools case: named explicitly, so it is not substituted."""
    monkeypatch.setattr("anycorn.protocol.h11.h11_available", lambda: False)
    config = Config()
    config.http_parser = "h11"

    with pytest.raises(RuntimeError, match="h11 is not installed"):
        _make_connection(config)


def test_auto_falls_back_to_httptools_without_h11(monkeypatch: pytest.MonkeyPatch) -> None:
    """h11 is preferred but not required: auto takes httptools when it is alone."""
    monkeypatch.setattr("anycorn.protocol.h11.h11_available", lambda: False)
    config = Config()
    config.http_parser = "auto"

    assert isinstance(_make_connection(config), HttpToolsConnection)


def test_with_neither_parser_installed_it_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anycorn.protocol.h11.h11_available", lambda: False)
    monkeypatch.setattr("anycorn.protocol.h11.httptools_available", lambda: False)
    config = Config()
    config.http_parser = "auto"

    with pytest.raises(RuntimeError, match=r"no HTTP/1\.1 parser is installed"):
        _make_connection(config)
