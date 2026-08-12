"""Tests for H11 protocol implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, call

import anyio
import h11
import pytest

import anycorn.protocol.h11
from anycorn.config import Config
from anycorn.events import Closed, RawData, SendFile, Updated
from anycorn.protocol.events import (
    Body,
    Data,
    EndBody,
    EndData,
    InformationalResponse,
    Request,
    Response,
    StreamClosed,
    Trailers,
    ZeroCopySend,
)
from anycorn.protocol.h11 import H2CProtocolRequiredError, H2ProtocolAssumedError, H11Protocol
from anycorn.protocol.http_stream import HTTPStream
from anycorn.protocol.ws_stream import WSStream
from anycorn.typing import ConnectionState
from anycorn.typing import Event as IOEvent
from anycorn.worker_context import EventWrapper
from tests.helpers import capture_logs

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


BASIC_HEADERS = [("Host", "anycorn"), ("Connection", "close")]


async def _handle_then_set(protocol: H11Protocol, data: bytes, done: anyio.Event) -> None:
    await protocol.handle(RawData(data=data))
    done.set()


@pytest.fixture(name="protocol")
async def _protocol(monkeypatch: MonkeyPatch) -> H11Protocol:
    MockHTTPStream = Mock()  # noqa: N806
    MockHTTPStream.return_value = AsyncMock(spec=HTTPStream)
    monkeypatch.setattr(anycorn.protocol.h11, "HTTPStream", MockHTTPStream)
    context = Mock()
    context.event_class.return_value = AsyncMock(spec=IOEvent)
    context.mark_request = AsyncMock()
    context.terminate = context.event_class()
    context.terminated = context.event_class()
    context.terminated.is_set.return_value = False
    return H11Protocol(
        AsyncMock(),
        Config(),
        context,
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
    )


@pytest.mark.anyio
async def test_protocol_send_response(protocol: H11Protocol) -> None:
    await protocol.stream_send(Response(stream_id=1, status_code=201, headers=[]))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert (
        protocol.send.call_args_list  # type: ignore[attr-defined]
        == [
            call(
                RawData(
                    data=(
                        b"HTTP/1.1 201 \r\ndate: Thu, 01 Jan 1970 01:23:20 GMT\r\n"
                        b"server: anycorn-h11\r\nConnection: close\r\n\r\n"
                    )
                )
            )
        ]
    )


@pytest.mark.anyio
async def test_protocol_preserve_headers(protocol: H11Protocol) -> None:
    await protocol.stream_send(
        Response(stream_id=1, status_code=201, headers=[(b"X-Special", b"Value")])
    )
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert (
        protocol.send.call_args_list  # type: ignore[attr-defined]
        == [
            call(
                RawData(
                    data=(
                        b"HTTP/1.1 201 \r\nX-Special: Value\r\n"
                        b"date: Thu, 01 Jan 1970 01:23:20 GMT\r\n"
                        b"server: anycorn-h11\r\nConnection: close\r\n\r\n"
                    )
                )
            )
        ]
    )


@pytest.mark.anyio
async def test_protocol_send_data(protocol: H11Protocol) -> None:
    await protocol.stream_send(Data(stream_id=1, data=b"hello"))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert protocol.send.call_args_list == [call(RawData(data=b"hello"))]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_protocol_zerocopysend_content_length(protocol: H11Protocol) -> None:
    """A Content-Length body becomes a bare SendFile - no framing bytes around it."""
    await protocol.stream_send(
        Response(stream_id=1, status_code=200, headers=[(b"content-length", b"10")])
    )
    protocol.send.reset_mock()  # type: ignore[attr-defined]
    await protocol.stream_send(ZeroCopySend(stream_id=1, file=7, offset=3, count=10))
    # h11 kept the framing (decremented the Content-Length) but handed back the file.
    assert protocol.send.call_args_list == [call(SendFile(file=7, offset=3, count=10))]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_protocol_zerocopysend_chunked_wraps_with_framing(protocol: H11Protocol) -> None:
    """Without a Content-Length h11 chunks, so the SendFile is wrapped by chunk framing."""
    # A keep-alive HTTP/1.1 request, so h11 must chunk rather than close-delimit the body.
    await protocol.handle(RawData(data=b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"))
    await protocol.stream_send(Response(stream_id=1, status_code=200, headers=[]))
    protocol.send.reset_mock()  # type: ignore[attr-defined]
    await protocol.stream_send(ZeroCopySend(stream_id=1, file=7, offset=0, count=10))
    assert protocol.send.call_args_list == [  # type: ignore[attr-defined]
        call(RawData(data=b"a\r\n")),  # chunk size, 10 in hex
        call(SendFile(file=7, offset=0, count=10)),
        call(RawData(data=b"\r\n")),
    ]


@pytest.mark.anyio
async def test_protocol_send_body(protocol: H11Protocol) -> None:
    await protocol.handle(
        RawData(data=b"GET / HTTP/1.1\r\nHost: anycorn\r\nConnection: close\r\n\r\n")
    )
    await protocol.stream_send(
        Response(stream_id=1, status_code=200, headers=[(b"content-length", b"5")])
    )
    await protocol.stream_send(Body(stream_id=1, data=b"hello"))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert protocol.send.call_args_list == [  # type: ignore[attr-defined]
        call(Updated(idle=False)),
        call(
            RawData(
                data=b"HTTP/1.1 200 \r\ncontent-length: 5\r\ndate: Thu, 01 Jan 1970 01:23:20 GMT\r\nserver: anycorn-h11\r\nConnection: close\r\n\r\n"  # noqa: E501
            )
        ),
        call(RawData(data=b"hello")),
    ]


@pytest.mark.anyio
async def test_protocol_keep_alive_max_requests(protocol: H11Protocol) -> None:
    data = b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"
    protocol.config.keep_alive_max_requests = 0
    await protocol.handle(RawData(data=data))
    await protocol.stream_send(Response(stream_id=1, status_code=200, headers=[]))
    await protocol.stream_send(EndBody(stream_id=1))
    await protocol.stream_send(StreamClosed(stream_id=1))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert protocol.send.call_args_list[3] == call(Closed())  # type: ignore[attr-defined]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("keep_alive", "expected"),
    [(True, Updated(idle=True)), (False, Closed())],
)
async def test_protocol_send_stream_closed(
    keep_alive: bool,  # noqa: FBT001
    expected: Any,  # noqa: ANN401
    protocol: H11Protocol,
) -> None:
    data = b"GET / HTTP/1.1\r\nHost: anycorn\r\n"
    if keep_alive:
        data += b"\r\n"
    else:
        data += b"Connection: close\r\n\r\n"
    await protocol.handle(RawData(data=data))
    await protocol.stream_send(Response(stream_id=1, status_code=200, headers=[]))
    await protocol.stream_send(EndBody(stream_id=1))
    await protocol.stream_send(StreamClosed(stream_id=1))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert protocol.send.call_args_list[3] == call(expected)  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_protocol_instant_recycle(protocol: H11Protocol) -> None:
    # This test task acts as the asgi app, spawned tasks act as the
    # server.
    data = b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"
    # This test requires a real event as the handling should pause on
    # the instant receipt
    protocol.can_read = EventWrapper()

    async with anyio.create_task_group() as task_group:
        handled = anyio.Event()
        task_group.start_soon(_handle_then_set, protocol, data, handled)
        await anyio.wait_all_tasks_blocked()
        assert protocol.stream is not None
        assert handled.is_set()

        await protocol.stream_send(Response(stream_id=1, status_code=200, headers=[]))
        await protocol.stream_send(EndBody(stream_id=1))

        # The second request arrives whilst the first stream is still open, so
        # handling it must wait rather than drop it
        recycled = anyio.Event()
        task_group.start_soon(_handle_then_set, protocol, data, recycled)
        await anyio.wait_all_tasks_blocked()
        assert not recycled.is_set()

        await protocol.stream_send(StreamClosed(stream_id=1))
        await anyio.wait_all_tasks_blocked()
        # Should have recycled, i.e. a stream should exist
        assert protocol.stream is not None
        assert recycled.is_set()


@pytest.mark.anyio
async def test_protocol_send_end_data(protocol: H11Protocol) -> None:
    protocol.stream = AsyncMock()
    await protocol.stream_send(EndData(stream_id=1))
    assert protocol.stream is not None


@pytest.mark.anyio
async def test_protocol_send_informational_response_is_ignored(protocol: H11Protocol) -> None:
    """HTTP/1 has no separate informational-response event here, so it is dropped."""
    await protocol.stream_send(InformationalResponse(stream_id=1, status_code=103, headers=[]))
    protocol.send.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_protocol_send_ignores_an_unhandled_event(protocol: H11Protocol) -> None:
    """A stream event with no HTTP/1 handling (e.g. Trailers) falls through untouched."""
    await protocol.stream_send(Trailers(stream_id=1, headers=[]))
    protocol.send.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_handle_ignores_events_other_than_raw_data_and_closed(protocol: H11Protocol) -> None:
    """Handle only reacts to RawData and Closed; anything else is a no-op."""
    await protocol.handle(Updated(idle=True))
    protocol.send.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_close_stream_without_a_stream_is_a_no_op(protocol: H11Protocol) -> None:
    """Closing when no stream was ever created must not fail."""
    assert protocol.stream is None
    await protocol._close_stream()  # nothing to close


@pytest.mark.anyio
async def test_a_framing_error_after_response_started_only_closes(protocol: H11Protocol) -> None:
    """A malformed body chunk once we are past sending headers just closes the connection.

    The error-status-hint reply is only sent while a response can still begin (our_state
    is IDLE or SEND_RESPONSE). Here the response has already started, so the framing error
    that h11 raises on the bad chunk is answered with a bare Closed, not a 4xx.
    """
    await protocol.handle(
        RawData(data=b"POST / HTTP/1.1\r\nHost: anycorn\r\nTransfer-Encoding: chunked\r\n\r\n")
    )
    # The app starts responding, advancing our_state past SEND_RESPONSE.
    await protocol.stream_send(
        Response(stream_id=1, status_code=200, headers=[(b"content-length", b"0")])
    )
    assert protocol.connection.our_state not in {h11.IDLE, h11.SEND_RESPONSE}
    protocol.send.reset_mock()  # type: ignore[attr-defined]

    # A chunk size that is not hex is a framing error h11 raises on.
    await protocol.handle(RawData(data=b"NOTHEX\r\n"))

    assert protocol.send.call_args_list == [call(Closed())]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_zerocopysend_is_swallowed_once_the_peer_has_errored(protocol: H11Protocol) -> None:
    """A zero-copy send after the client wedged the connection into ERROR is dropped.

    A malformed request drives h11's their_state to ERROR; a later send then raises
    LocalProtocolError, which is swallowed rather than propagated - exactly as the
    ordinary _send_h11_event path does.
    """
    await protocol.handle(RawData(data=b"broken nonsense\r\n\r\n"))
    assert protocol.connection.their_state is h11.ERROR
    # Must not raise despite the connection being unusable.
    await protocol.stream_send(ZeroCopySend(stream_id=1, file=7, offset=0, count=10))


@pytest.mark.anyio
async def test_protocol_handle_closed(protocol: H11Protocol) -> None:
    await protocol.handle(
        RawData(data=b"GET / HTTP/1.1\r\nHost: anycorn\r\nConnection: close\r\n\r\n")
    )
    stream = protocol.stream
    await protocol.handle(Closed())
    stream.handle.assert_called()  # type: ignore[attr-defined]
    assert stream.handle.call_args_list == [  # type: ignore[attr-defined]
        call(
            Request(
                stream_id=1,
                headers=[(b"host", b"anycorn"), (b"connection", b"close")],
                http_version="1.1",
                method="GET",
                raw_path=b"/",
                state=ConnectionState({}),
            )
        ),
        call(EndBody(stream_id=1)),
        call(StreamClosed(stream_id=1)),
    ]


@pytest.mark.anyio
async def test_protocol_handle_request(protocol: H11Protocol) -> None:
    client = h11.Connection(h11.CLIENT)
    await protocol.handle(
        RawData(data=client.send(h11.Request(method="GET", target="/?a=b", headers=BASIC_HEADERS)))
    )
    protocol.stream.handle.assert_called()  # type: ignore[attr-defined]
    assert protocol.stream.handle.call_args_list == [  # type: ignore[attr-defined]
        call(
            Request(
                stream_id=1,
                headers=[(b"host", b"anycorn"), (b"connection", b"close")],
                http_version="1.1",
                method="GET",
                raw_path=b"/?a=b",
                state=ConnectionState({}),
            )
        ),
        call(EndBody(stream_id=1)),
    ]


@pytest.mark.anyio
async def test_protocol_handle_request_with_raw_headers(protocol: H11Protocol) -> None:
    protocol.config.h11_pass_raw_headers = True
    client = h11.Connection(h11.CLIENT)
    headers = [*BASIC_HEADERS, ("FOO_BAR", "foobar")]
    await protocol.handle(
        RawData(data=client.send(h11.Request(method="GET", target="/?a=b", headers=headers)))
    )
    protocol.stream.handle.assert_called()  # type: ignore[attr-defined]
    assert protocol.stream.handle.call_args_list == [  # type: ignore[attr-defined]
        call(
            Request(
                stream_id=1,
                headers=[
                    (b"Host", b"anycorn"),
                    (b"Connection", b"close"),
                    (b"FOO_BAR", b"foobar"),
                ],
                http_version="1.1",
                method="GET",
                raw_path=b"/?a=b",
                state=ConnectionState({}),
            )
        ),
        call(EndBody(stream_id=1)),
    ]


@pytest.mark.anyio
async def test_protocol_handle_protocol_error(protocol: H11Protocol) -> None:
    await protocol.handle(RawData(data=b"broken nonsense\r\n\r\n"))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert (
        protocol.send.call_args_list  # type: ignore[attr-defined]
        == [
            call(
                RawData(
                    data=b"HTTP/1.1 400 \r\ncontent-length: 0\r\nconnection: close\r\n"
                    b"date: Thu, 01 Jan 1970 01:23:20 GMT\r\nserver: anycorn-h11\r\n\r\n"
                )
            ),
            call(RawData(data=b"")),
            call(Closed()),
        ]
    )


@pytest.mark.anyio
async def test_protocol_handle_send_client_error(protocol: H11Protocol) -> None:
    client = h11.Connection(h11.CLIENT)
    await protocol.handle(
        RawData(data=client.send(h11.Request(method="GET", target="/?a=b", headers=BASIC_HEADERS)))
    )
    await protocol.handle(RawData(data=b"some body"))
    # This next line should not cause an error
    await protocol.stream_send(Response(stream_id=1, status_code=200, headers=[]))


@pytest.mark.anyio
async def test_protocol_handle_pipelining(protocol: H11Protocol) -> None:
    protocol.can_read.wait.side_effect = Exception("pipelining")
    with pytest.raises(Exception, match="pipelining"):
        await protocol.handle(
            RawData(
                data=b"GET / HTTP/1.1\r\nHost: anycorn\r\nConnection: keep-alive\r\n\r\n"
                b"GET / HTTP/1.1\r\nHost: anycorn\r\nConnection: close\r\n\r\n"
            )
        )
    protocol.can_read.clear.assert_called()
    protocol.can_read.wait.assert_called()


@pytest.mark.anyio
async def test_protocol_handle_continue_request(protocol: H11Protocol) -> None:
    client = h11.Connection(h11.CLIENT)
    await protocol.handle(
        RawData(
            data=client.send(
                h11.Request(
                    method="POST",
                    target="/?a=b",
                    headers=[
                        *BASIC_HEADERS,
                        ("transfer-encoding", "chunked"),
                        ("expect", "100-continue"),
                    ],
                )
            )
        )
    )
    assert protocol.send.call_args[0][0] == RawData(  # type: ignore[attr-defined]
        data=b"HTTP/1.1 100 \r\ndate: Thu, 01 Jan 1970 01:23:20 GMT\r\nserver: anycorn-h11\r\n\r\n"
    )


@pytest.mark.anyio
async def test_protocol_handle_max_incomplete(monkeypatch: MonkeyPatch) -> None:
    config = Config()
    config.h11_max_incomplete_size = 5
    MockHTTPStream = AsyncMock()  # noqa: N806
    MockHTTPStream.return_value = AsyncMock(spec=HTTPStream)
    monkeypatch.setattr(anycorn.protocol.h11, "HTTPStream", MockHTTPStream)
    context = Mock()
    context.event_class.return_value = AsyncMock(spec=IOEvent)
    protocol = H11Protocol(
        AsyncMock(),
        config,
        context,
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
    )
    await protocol.handle(RawData(data=b"GET / HTTP/1.1\r\nHost: anycorn\r\n"))
    protocol.send.assert_called()  # type: ignore[attr-defined]
    assert (
        protocol.send.call_args_list  # type: ignore[attr-defined]
        == [
            call(
                RawData(
                    data=b"HTTP/1.1 431 \r\ncontent-length: 0\r\nconnection: close\r\n"
                    b"date: Thu, 01 Jan 1970 01:23:20 GMT\r\nserver: anycorn-h11\r\n\r\n"
                )
            ),
            call(RawData(data=b"")),
            call(Closed()),
        ]
    )


@pytest.mark.anyio
async def test_protocol_handle_h2c_upgrade(protocol: H11Protocol) -> None:
    with pytest.raises(H2CProtocolRequiredError) as exc_info:
        await protocol.handle(
            RawData(
                data=(
                    b"GET / HTTP/1.1\r\nHost: anycorn\r\n"
                    b"upgrade: h2c\r\nhttp2-settings: abcd\r\n\r\nbbb"
                )
            )
        )
    assert (
        protocol.send.call_args_list  # type: ignore[attr-defined]
        == [
            call(Updated(idle=False)),
            call(
                RawData(
                    b"HTTP/1.1 101 \r\n"
                    b"date: Thu, 01 Jan 1970 01:23:20 GMT\r\n"
                    b"server: anycorn-h11\r\n"
                    b"connection: upgrade\r\n"
                    b"upgrade: h2c\r\n"
                    b"\r\n"
                )
            ),
        ]
    )
    assert exc_info.value.data == b"bbb"
    assert exc_info.value.headers == [
        (b":method", b"GET"),
        (b":path", b"/"),
        (b":authority", b"anycorn"),
        (b"host", b"anycorn"),
        (b"upgrade", b"h2c"),
        (b"http2-settings", b"abcd"),
    ]
    assert exc_info.value.settings == "abcd"


@pytest.mark.anyio
async def test_protocol_handle_h2_prior(protocol: H11Protocol) -> None:
    with pytest.raises(H2ProtocolAssumedError) as exc_info:
        await protocol.handle(RawData(data=b"PRI * HTTP/2.0\r\n\r\nbbb"))

    assert exc_info.value.data == b"PRI * HTTP/2.0\r\n\r\nbbb"


@pytest.mark.anyio
async def test_protocol_handle_data_post_response(protocol: H11Protocol) -> None:
    await protocol.handle(
        RawData(data=b"POST / HTTP/1.1\r\nHost: anycorn\r\nContent-Length: 4\r\n\r\n")
    )
    await protocol.stream_send(Response(stream_id=1, status_code=201, headers=[]))
    await protocol.stream_send(EndBody(stream_id=1))
    await protocol.handle(RawData(data=b"abcd"))


@pytest.mark.anyio
async def test_protocol_handle_data_post_end(protocol: H11Protocol) -> None:
    await protocol.handle(
        RawData(data=b"POST / HTTP/1.1\r\nHost: anycorn\r\nContent-Length: 10\r\n\r\n")
    )
    await protocol.stream_send(Response(stream_id=1, status_code=201, headers=[]))
    await protocol.stream_send(EndBody(stream_id=1))
    # Key is that this doesn't error
    await protocol.handle(RawData(data=b"abcdefghij"))


@pytest.mark.anyio
async def test_protocol_handle_data_post_close(protocol: H11Protocol) -> None:
    await protocol.handle(
        RawData(data=b"POST / HTTP/1.1\r\nHost: anycorn\r\nContent-Length: 10\r\n\r\n")
    )
    await protocol.stream_send(StreamClosed(stream_id=1))
    assert protocol.stream is None
    # Key is that this doesn't error
    await protocol.handle(RawData(data=b"abcdefghij"))


@pytest.mark.anyio
async def test_protocol_handle_data_after_websocket_upgrade(protocol: H11Protocol) -> None:
    """Trailing data on a websocket upgrade must not crash the worker.

    The bytes after the handshake arrive as a Data event before the app has
    accepted the connection - while WSStream still has no wsproto connection
    object to decode them with. Reading self.connection there raised
    AttributeError and took the whole worker down with it.

    They are held until the handshake resolves now, rather than answered, so
    nothing is sent here: this app never accepts. What matters for the regression
    is that handling them raises nothing.

    https://github.com/pgjones/hypercorn/issues/225
    """
    request = (
        b"GET / HTTP/1.1\r\n"
        b"Host: anycorn\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: websocket\r\n"
        b"Sec-WebSocket-Version: 13\r\n"
        b"Sec-WebSocket-Key: bKdPyn3u98cTfZJSh4TNeQ==\r\n"
        b"\r\n"
        b"x"
    )

    await protocol.handle(RawData(data=request))  # must not raise

    sent = b"".join(
        event.data
        for (event, *_), _ in protocol.send.call_args_list  # type: ignore[attr-defined]
        if isinstance(event, RawData)
    )
    assert sent == b"", "the trailing byte should be held, not answered"
    assert isinstance(protocol.stream, WSStream)
    assert protocol.stream.pre_accept_data == b"x"


@pytest.mark.anyio
async def test_protocol_logs_a_rejected_request(protocol: H11Protocol) -> None:
    """A request h11 rejects (e.g. 431 for oversized headers) is logged.

    The status was already surfaced correctly via error_status_hint; what was missing
    is any record of why the request was turned away, which made the rejection opaque.

    https://github.com/pgjones/hypercorn/issues/157
    """
    logs = capture_logs(protocol.config)

    await protocol.handle(RawData(data=b"broken nonsense\r\n\r\n"))

    # The real error log ran and recorded why, which is the point of the fix; the
    # mock this replaces asserted only that something had been called.
    assert len(logs.error) == 1
    assert "Rejecting request" in logs.error[0]
    assert "(400)" in logs.error[0]


@pytest.mark.anyio
async def test_protocol_does_not_recycle_once_terminated(protocol: H11Protocol) -> None:
    """A worker shutting down closes after its response rather than taking another request.

    The mirror of test_protocol_instant_recycle, which drives this very exchange on a
    running worker and has the waiting second request served the moment the first
    stream closes. Only terminated differs here, and it must be Closed that follows
    rather than the connection being handed back for reuse.
    """
    protocol.context.terminated.is_set.return_value = True  # type: ignore[attr-defined]
    data = b"GET / HTTP/1.1\r\nHost: anycorn\r\n\r\n"
    protocol.can_read = EventWrapper()

    async with anyio.create_task_group() as task_group:
        handled = anyio.Event()
        task_group.start_soon(_handle_then_set, protocol, data, handled)
        await anyio.wait_all_tasks_blocked()
        assert protocol.stream is not None

        await protocol.stream_send(Response(stream_id=1, status_code=200, headers=[]))
        await protocol.stream_send(EndBody(stream_id=1))

        # Queued behind the first, exactly as in the recycling test
        recycled = anyio.Event()
        task_group.start_soon(_handle_then_set, protocol, data, recycled)
        await anyio.wait_all_tasks_blocked()

        await protocol.stream_send(StreamClosed(stream_id=1))
        await anyio.wait_all_tasks_blocked()

        # No new cycle: the queued request is not taken up and no stream replaces the one
        # that closed
        assert not recycled.is_set()
        assert protocol.stream is None
        task_group.cancel_scope.cancel()  # release the request left waiting

    assert call(Closed()) in protocol.send.call_args_list
    assert call(Updated(idle=True)) not in protocol.send.call_args_list
