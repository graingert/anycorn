"""Tests for the httptools-backed connection.

The contract is not "behaves reasonably" but "behaves as h11 does", since
`H11Protocol` and everything above it reads the events and the bytes without
knowing which parser produced them. So most of this compares the two directly
rather than asserting literals, which would only record whatever this
implementation happened to do.
"""

from __future__ import annotations

import h11
import pytest

from anycorn.config import Config
from anycorn.protocol.h11 import _make_connection
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
            if event is h11.NEED_DATA or isinstance(
                event, (h11.EndOfMessage, h11.ConnectionClosed)
            ):
                break

    return events, b"".join(_encode(connection, event) for event in sends)


def _encode(connection: object, event: object) -> bytes:
    try:
        return connection.send(event)  # type: ignore[attr-defined]
    except h11.LocalProtocolError:
        return b"<LocalProtocolError>"


@pytest.mark.parametrize(
    ("request_bytes", "sends"),
    [
        # A response before any request has arrived, as an error response is
        (None, [h11.Response(status_code=201, headers=[]), h11.EndOfMessage()]),
        (
            GET,
            [
                h11.Response(status_code=200, headers=[(b"content-length", b"4")]),
                h11.Data(data=b"body"),
                h11.EndOfMessage(),
            ],
        ),
        # No framing given, so the connection has to choose one
        (
            GET,
            [h11.Response(status_code=200, headers=[]), h11.Data(data=b"body"), h11.EndOfMessage()],
        ),
        # HTTP/1.0 cannot be sent chunked, so it is delimited by closing
        (
            GET_10,
            [h11.Response(status_code=200, headers=[]), h11.Data(data=b"body"), h11.EndOfMessage()],
        ),
        # Framed as the equivalent GET, but no body written
        (HEAD, [h11.Response(status_code=200, headers=[]), h11.EndOfMessage()]),
        (GET, [h11.Response(status_code=204, headers=[]), h11.EndOfMessage()]),
        (GET, [h11.Response(status_code=304, headers=[]), h11.EndOfMessage()]),
        # The case the app wrote its header names in is preserved
        (
            GET,
            [h11.Response(status_code=201, headers=[(b"X-Special", b"Value")]), h11.EndOfMessage()],
        ),
        # An app that closes the connection itself must not get two of them
        (
            GET,
            [
                h11.Response(
                    status_code=200, headers=[(b"content-length", b"0"), (b"connection", b"close")]
                ),
                h11.EndOfMessage(),
            ],
        ),
        (
            POST,
            [
                h11.Response(status_code=200, headers=[(b"content-length", b"2")]),
                h11.Data(data=b"ok"),
                h11.EndOfMessage(),
            ],
        ),
        (
            b"POST / HTTP/1.1\r\nHost: a\r\nExpect: 100-continue\r\ncontent-length: 4\r\n\r\n",
            [h11.InformationalResponse(status_code=100, headers=[])],
        ),
    ],
)
def test_writes_the_same_bytes_as_h11(request_bytes: bytes | None, sends: list) -> None:
    """Byte for byte, so nothing downstream can tell which parser answered."""
    _, mine = _drive(HttpToolsConnection(16 * 1024), request_bytes, sends)
    _, theirs = _drive(h11.Connection(h11.SERVER), request_bytes, sends)

    assert mine == theirs


@pytest.mark.parametrize("request_bytes", [GET, GET_10, HEAD, POST])
def test_reads_the_same_events_as_h11(request_bytes: bytes) -> None:
    """The event stream has to match too, since H11Protocol switches on its types."""
    mine, _ = _drive(HttpToolsConnection(16 * 1024), request_bytes, [])
    theirs, _ = _drive(h11.Connection(h11.SERVER), request_bytes, [])

    assert [type(event).__name__ for event in mine] == [type(event).__name__ for event in theirs]
    assert mine[0].method == theirs[0].method
    assert mine[0].target == theirs[0].target
    assert mine[0].http_version == theirs[0].http_version
    assert list(mine[0].headers) == list(theirs[0].headers)


def test_a_second_pipelined_request_pauses_rather_than_merging() -> None:
    """Httptools reads straight on into the next request; h11 stops and pauses.

    Left alone the two get folded together, which surfaces as a duplicate-header
    error rather than as pipelining - and the second request is lost.
    """
    keep_alive = b"GET /first HTTP/1.1\r\nHost: a\r\nConnection: keep-alive\r\n\r\n"
    connection = HttpToolsConnection(16 * 1024)
    connection.receive_data(keep_alive + b"GET /second HTTP/1.1\r\nHost: a\r\n\r\n")

    assert connection.next_event().target == b"/first"
    assert isinstance(connection.next_event(), h11.EndOfMessage)
    assert connection.next_event() is h11.PAUSED

    # Recycling needs both sides finished, which is what H11Protocol waits for
    connection.send(h11.Response(status_code=200, headers=[(b"content-length", b"0")]))
    connection.send(h11.EndOfMessage())
    connection.start_next_cycle()

    assert connection.next_event().target == b"/second"


def test_data_after_a_closing_request_is_an_error() -> None:
    """Which is what h11 answers, rather than treating it as the next request."""
    connection = HttpToolsConnection(16 * 1024)
    connection.receive_data(b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\r\n")
    assert connection.next_event().target == b"/"
    assert isinstance(connection.next_event(), h11.EndOfMessage)

    connection.receive_data(b"some body")

    with pytest.raises(h11.RemoteProtocolError):
        connection.next_event()


def test_an_oversized_header_block_is_rejected() -> None:
    """The limit h11 enforces via max_incomplete_event_size, with h11's status."""
    connection = HttpToolsConnection(64)
    connection.receive_data(b"GET / HTTP/1.1\r\nHost: a\r\nX-Long: " + b"a" * 200)

    with pytest.raises(h11.RemoteProtocolError) as exc_info:
        connection.next_event()

    assert exc_info.value.error_status_hint == 431  # noqa: PLR2004


def test_the_http2_preface_is_handed_over_as_a_request() -> None:
    """Httptools rejects it outright, where h11 parses it and lets the caller decide.

    H11Protocol looks for exactly this to spot a client speaking HTTP/2 with prior
    knowledge, so the preface has to arrive as a request rather than as a 400.
    """
    connection = HttpToolsConnection(16 * 1024)
    connection.receive_data(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")

    event = connection.next_event()

    assert event.method == b"PRI"
    assert event.target == b"*"
    assert event.http_version == b"2.0"
    assert connection.trailing_data[0] == b"SM\r\n\r\n"


@pytest.mark.parametrize(
    ("parser", "expected"),
    [("h11", h11.Connection), ("httptools", HttpToolsConnection), ("auto", HttpToolsConnection)],
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


def test_auto_falls_back_to_h11_without_httptools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto is a preference, so it takes what is there."""
    monkeypatch.setattr("anycorn.protocol.h11.httptools_available", lambda: False)
    config = Config()
    config.http_parser = "auto"

    assert isinstance(_make_connection(config), h11.Connection)
