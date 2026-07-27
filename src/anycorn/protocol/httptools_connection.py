"""An h11.Connection work-alike that parses with httptools.

httptools wraps llhttp, the parser Node uses, and is markedly faster than h11 at
reading requests - but it only reads them. Everything else h11 does for us, the
state machine and the response encoding, has to be supplied here.

What this deliberately does *not* do is add a second HTTP/1.1 protocol
implementation. `H11Protocol` already talks to its connection through a small
interface - it holds `h11.Connection | H11WSConnection` and swaps one for the
other on a websocket upgrade - so a third implementation of that interface leaves
keep-alive, the websocket upgrade, the h2c upgrade and 100-continue handling
exactly where they are, with one code path rather than two.

For the same reason the events emitted here are h11's own, and the states are
h11's sentinels: `H11Protocol` compares against `h11.DONE` and reads
`event.headers`, and none of that should have to care which parser produced them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h11

try:
    import httptools
except ImportError:  # pragma: no cover - exercised by not installing the extra
    httptools = None  # type: ignore[assignment]


def httptools_available() -> bool:
    """Return True when the optional httptools dependency can be used."""
    return httptools is not None


if TYPE_CHECKING:
    from anycorn.typing import H11SendableEvent

# What h11 reports for a request whose framing we cannot honour
_BAD_REQUEST = 400
_REQUEST_HEADER_FIELDS_TOO_LARGE = 431

# Statuses that carry no body however the app frames them (RFC 9110 6.4.1)
_NO_BODY_STATUSES = {204, 304}
_INFORMATIONAL = 200
# What a client speaking HTTP/2 with prior knowledge opens with (RFC 9113 3.4)
_H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\n"


class _Callbacks:
    """Receives httptools' parse callbacks and records what they report.

    Kept apart from the connection so the parser cannot see - and so cannot be
    made to drive - anything but the fields it is meant to fill in.
    """

    def __init__(self) -> None:
        self.url: bytes = b""
        self.headers: list[tuple[bytes, bytes]] = []
        self.body = bytearray()
        self.request_complete = False
        self.headers_complete = False
        self.messages_begun = 0
        self.expects_continue = False
        self.wants_close = False

    def on_url(self, url: bytes) -> None:
        self.url += url

    def on_header(self, name: bytes, value: bytes) -> None:
        self.headers.append((name, value))
        lowered = name.lower()
        if lowered == b"expect" and value.lower() == b"100-continue":
            self.expects_continue = True
        elif lowered == b"connection" and b"close" in value.lower():
            # httptools' should_keep_alive() reports True for an HTTP/1.1 request
            # carrying this, so the header has to be read here to honour it
            self.wants_close = True

    def on_headers_complete(self) -> None:
        self.headers_complete = True

    def on_body(self, body: bytes) -> None:
        self.body.extend(body)

    def on_message_begin(self) -> None:
        self.messages_begun += 1

    def on_message_complete(self) -> None:
        self.request_complete = True


class HttpToolsConnection:
    """The slice of `h11.Connection` that `H11Protocol` uses, parsed by httptools."""

    def __init__(self, max_incomplete_event_size: int) -> None:
        """Start a connection able to read one request at a time."""
        self.our_state: Any = h11.IDLE
        self.their_state: Any = h11.IDLE
        self.they_are_waiting_for_100_continue = False
        self.trailing_data: tuple[bytes, bool] = (b"", False)

        self._max_incomplete_event_size = max_incomplete_event_size
        self._method = b""
        self._their_http_version: bytes | None = None
        self._keep_alive = True
        # Set once a response has been framed, so Data and EndOfMessage know how
        self._chunked_response = False
        self._response_has_body = True
        self._start_cycle()

    def _start_cycle(self) -> None:
        self._callbacks = _Callbacks()
        self._parser = httptools.HttpRequestParser(self._callbacks)
        self._buffer = bytearray()
        self._sent_request = False
        self._sent_end_of_message = False
        self._upgrade_data = b""
        self._pipelined = b""
        self._error: h11.RemoteProtocolError | None = None

    # -- receiving -----------------------------------------------------------

    def receive_data(self, data: bytes) -> None:
        """Feed *data* to the parser, or record the peer having hung up."""
        if data == b"":
            self.their_state = h11.DONE if self._callbacks.request_complete else h11.ERROR
            return

        self._buffer.extend(data)
        if self._h2_preface_pending():
            return

        try:
            self._parser.feed_data(data)
        except httptools.HttpParserUpgrade as upgrade:
            # An upgrade (websocket, h2c) leaves whatever follows the request for
            # whoever takes the connection over, exactly as h11's trailing_data
            # does. The offset httptools reports is into this chunk, not the whole
            # buffer, so it has to be rebased before it means anything.
            offset = upgrade.args[0] if upgrade.args else len(data)
            self._upgrade_data = bytes(self._buffer[len(self._buffer) - len(data) + offset :])
        except httptools.HttpParserError as error:
            if self._callbacks.request_complete:
                self._handle_data_past_request()
            else:
                # Held rather than raised: H11Protocol guards next_event(), not this,
                # and h11 reports parse errors from there too
                self._error = _protocol_error(error)
        else:
            if self._callbacks.messages_begun > 1:
                self._handle_data_past_request()
            if not self._callbacks.headers_complete and len(self._buffer) > (
                self._max_incomplete_event_size
            ):
                self._error = _remote_error(
                    "Request header fields too large", _REQUEST_HEADER_FIELDS_TOO_LARGE
                )

    def _handle_data_past_request(self) -> None:
        """Deal with bytes that arrived after a request had already parsed.

        Which of the two h11 does depends on whether the connection is staying
        open: on a keep-alive connection more bytes are the next pipelined
        request, and on one already closing they are data the peer had no business
        sending - "Got data when expecting EOF".
        """
        if self._keep_alive:
            self._split_pipelined()
        else:
            self._error = _remote_error("Got data when expecting EOF", _BAD_REQUEST)

    def _split_pipelined(self) -> None:
        """Re-parse just the first request, holding the rest for the next cycle.

        httptools reads straight on into a second pipelined request and folds it
        into the first, which surfaces as a duplicate-header error rather than as
        pipelining. h11 hands out one request and then PAUSEs, so find the boundary
        and do the same. Byte at a time, which is slow - but only ever runs on a
        connection that actually pipelined, and only over one request's worth.
        """
        buffer = bytes(self._buffer)
        callbacks = _Callbacks()
        parser = httptools.HttpRequestParser(callbacks)
        for index in range(len(buffer)):
            parser.feed_data(buffer[index : index + 1])
            if callbacks.request_complete:
                self._callbacks = callbacks
                self._parser = parser
                self._pipelined = buffer[index + 1 :]
                self._buffer = bytearray(buffer[: index + 1])
                return

    def _h2_preface_pending(self) -> bool:
        """Return True while the buffer could still be an HTTP/2 prior-knowledge start.

        httptools rejects `PRI * HTTP/2.0` outright, so a client opening with the
        preface would be answered 400 rather than upgraded. h11 parses it as an
        ordinary request and leaves _check_protocol to notice, so hand it the same
        thing rather than teaching the caller about parsers.
        """
        return _H2_PREFACE.startswith(bytes(self._buffer[: len(_H2_PREFACE)]))

    def next_event(self) -> Any:  # noqa: ANN401, PLR0911
        """Return the next h11 event, or NEED_DATA when the parser wants more."""
        callbacks = self._callbacks
        if self._error is not None:
            self.their_state = h11.ERROR
            error, self._error = self._error, None
            raise error
        if self.their_state is h11.ERROR:
            return h11.ConnectionClosed()

        if bytes(self._buffer[: len(_H2_PREFACE)]) == _H2_PREFACE:
            self._sent_request = True
            self.their_state = h11.SEND_BODY
            self.trailing_data = (bytes(self._buffer[len(_H2_PREFACE) :]), False)
            return h11.Request(method=b"PRI", target=b"*", headers=[], http_version=b"2.0")
        if self._h2_preface_pending():
            return h11.NEED_DATA

        if callbacks.headers_complete and not self._sent_request:
            self._sent_request = True
            self._method = self._parser.get_method().upper()
            self._their_http_version = self._parser.get_http_version().encode("ascii")
            self._keep_alive = bool(self._parser.should_keep_alive()) and not callbacks.wants_close
            self.their_state = h11.SEND_BODY
            self.trailing_data = (self._pipelined or self._upgrade_data, False)
            # Raised here rather than on receipt: h11 only knows once it has handed
            # the request over, so the 100 follows the request rather than leading it
            self.they_are_waiting_for_100_continue = (
                callbacks.expects_continue and not callbacks.request_complete
            )
            return h11.Request(
                method=self._method,
                target=callbacks.url,
                headers=callbacks.headers,
                http_version=self._their_http_version,
            )

        if not self._sent_request:
            return h11.NEED_DATA

        if callbacks.body:
            data = bytes(callbacks.body)
            callbacks.body.clear()
            return h11.Data(data=data)

        if callbacks.request_complete and not self._sent_end_of_message:
            self._sent_end_of_message = True
            self.they_are_waiting_for_100_continue = False
            self.their_state = h11.DONE
            return h11.EndOfMessage()

        if self._pipelined:
            # Another request is already buffered; h11 pauses here rather than
            # reading ahead, and H11Protocol waits on can_read for the recycle
            return h11.PAUSED

        return h11.NEED_DATA

    # -- sending -------------------------------------------------------------

    def send(self, event: H11SendableEvent) -> bytes:  # noqa: PLR0911
        """Encode *event* the way h11 would, and advance our side of the state."""
        if isinstance(event, h11.InformationalResponse):
            self.they_are_waiting_for_100_continue = False
            return self._encode_head(event.status_code, event.headers, informational=True)
        if isinstance(event, h11.Response):
            if self.our_state not in {h11.IDLE, h11.SEND_RESPONSE}:
                msg = f"Cannot send a Response in state {self.our_state}"
                raise h11.LocalProtocolError(msg)
            self.our_state = h11.SEND_BODY
            return self._encode_head(event.status_code, event.headers, informational=False)
        if isinstance(event, h11.Data):
            if not self._response_has_body:
                return b""
            if self._chunked_response:
                return b"%x\r\n%s\r\n" % (len(event.data), bytes(event.data))
            return bytes(event.data)
        if isinstance(event, h11.EndOfMessage):
            if self.our_state is not h11.SEND_BODY:
                msg = f"Cannot send EndOfMessage in state {self.our_state}"
                raise h11.LocalProtocolError(msg)
            self.our_state = h11.DONE if self._keep_alive else h11.MUST_CLOSE
            if self._chunked_response and self._response_has_body:
                return b"0\r\n\r\n"
            return b""
        msg = f"Cannot send {type(event).__name__}"
        raise h11.LocalProtocolError(msg)

    def _encode_head(
        self,
        status_code: int,
        headers: Any,  # noqa: ANN401 - h11.Headers: pairs, plus raw_items()
        *,
        informational: bool,
    ) -> bytes:
        """Write a response head, choosing framing exactly as h11 does.

        h11 sends no reason phrase and capitalises the two headers it adds itself,
        and the tests here assert those bytes; matching them keeps the choice of
        parser invisible to everything downstream.
        """
        lines = [b"HTTP/1.1 %d " % status_code]
        lines.extend(b"%s: %s" % (name, value) for name, value in _raw_items(headers))

        if informational:
            return b"\r\n".join([*lines, b"", b""])

        # HEAD is framed as the equivalent GET would be - the headers say what a
        # body would have looked like - but no body is written
        self._response_has_body = self._method != b"HEAD"
        self._chunked_response = False
        need_close = False

        if _framing_is_unknown_length(status_code, headers):
            if self._their_http_version is None or self._their_http_version < b"1.1":
                # Either no valid request arrived, so assume the worst, or the peer
                # is HTTP/1.0 and does not understand chunked: delimit by closing
                if self._method != b"HEAD":
                    need_close = True
            else:
                self._chunked_response = True
                lines.append(b"Transfer-Encoding: chunked")
        elif not _allows_body(status_code):
            self._response_has_body = False

        already_closing = any(
            name.lower() == b"connection" and b"close" in value.lower() for name, value in headers
        )
        if already_closing:
            self._keep_alive = False
        if (not self._keep_alive or need_close) and not already_closing:
            self._keep_alive = False
            lines.append(b"Connection: close")

        return b"\r\n".join([*lines, b"", b""])

    def start_next_cycle(self) -> None:
        """Ready the connection for the next request on it."""
        if self.our_state is not h11.DONE or self.their_state is not h11.DONE:
            msg = f"Cannot start a new cycle in states {self.our_state}, {self.their_state}"
            raise h11.LocalProtocolError(msg)
        self.our_state = h11.IDLE
        self.their_state = h11.IDLE
        self.they_are_waiting_for_100_continue = False
        pipelined, self._pipelined = self._pipelined, b""
        self._start_cycle()
        if pipelined:
            self.receive_data(pipelined)


def _raw_items(headers: Any) -> list[tuple[bytes, bytes]]:  # noqa: ANN401
    """Return the headers with the case the caller wrote, as h11's writer does."""
    raw_items = getattr(headers, "raw_items", None)
    return list(raw_items()) if raw_items is not None else list(headers)


def _allows_body(status_code: int) -> bool:
    """Return True where the status permits a body at all (RFC 9110 6.4.1)."""
    return not (status_code in _NO_BODY_STATUSES or status_code < _INFORMATIONAL)


def _framing_is_unknown_length(status_code: int, headers: Any) -> bool:  # noqa: ANN401
    """Return True when the app gave no framing for a response that may have a body."""
    if not _allows_body(status_code):
        return False
    return not any(name.lower() in {b"content-length", b"transfer-encoding"} for name, _ in headers)


def _remote_error(message: str, status_hint: int) -> h11.RemoteProtocolError:
    return h11.RemoteProtocolError(message, error_status_hint=status_hint)


def _protocol_error(error: Exception) -> h11.RemoteProtocolError:
    return _remote_error(str(error) or type(error).__name__, _BAD_REQUEST)
