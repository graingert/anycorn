"""Tests for the HTTP/2 protocol handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, call

import anyio
import pytest
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import ConnectionTerminated, PushedStreamReceived
from h2.settings import SettingCodes

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.events import Closed, Event, RawData, Updated
from anycorn.protocol.events import Body, Trailers
from anycorn.protocol.events import Event as StreamEvent
from anycorn.protocol.h2 import (
    BUFFER_HIGH_WATER,
    BufferCompleteError,
    H2Protocol,
    StreamBuffer,
)
from anycorn.task_group import TaskGroup
from anycorn.typing import ASGIReceiveCallable, ASGISendCallable, ConnectionState, Scope
from anycorn.worker_context import EventWrapper, WorkerContext


def _protocol(app: object = None, send: object = None) -> H2Protocol:
    return H2Protocol(
        ASGIWrapper(app) if app is not None else Mock(),  # type: ignore[arg-type]
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        send if send is not None else AsyncMock(),  # type: ignore[arg-type]
        None,
    )


def _client_request(client: H2Connection, stream_id: int = 1, *, path: bytes = b"/") -> bytes:
    client.send_headers(
        stream_id,
        [
            (b":method", b"GET"),
            (b":path", path),
            (b":authority", b"anycorn"),
            (b":scheme", b"https"),
        ],
        end_stream=True,
    )
    return client.data_to_send()


@pytest.mark.anyio
async def test_stream_buffer_push_and_pop() -> None:
    """The writer resumes once the buffer has drained, not once a chunk is taken.

    Resuming on the size of the chunk popped meant a zero-length pop - which is
    what _send_data does once the peer's flow control window is exhausted - let
    the writer on with the buffer still above the high water mark.
    """
    stream_buffer = StreamBuffer(EventWrapper)
    pushed = False

    async def _push_over_limit() -> None:
        nonlocal pushed
        await stream_buffer.push(b"a" * (BUFFER_HIGH_WATER + 1))
        pushed = True

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_push_over_limit)
        await anyio.wait_all_tasks_blocked()
        assert not pushed  # Blocked as over high water

        # Nothing could be sent, so nothing has drained and the writer stays put
        await stream_buffer.pop(0)
        await anyio.wait_all_tasks_blocked()
        assert not pushed

        # Drained some, but not to below the low water mark
        await stream_buffer.pop(BUFFER_HIGH_WATER // 4)
        await anyio.wait_all_tasks_blocked()
        assert not pushed

        # Now under it, so there is room for the writer to go on
        await stream_buffer.pop(BUFFER_HIGH_WATER)
        await anyio.wait_all_tasks_blocked()
        assert pushed


@pytest.mark.anyio
async def test_stream_buffer_drain() -> None:
    stream_buffer = StreamBuffer(EventWrapper)
    await stream_buffer.push(b"a" * 10)
    drained = False

    async def _drain() -> None:
        nonlocal drained
        await stream_buffer.drain()
        drained = True

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_drain)
        await anyio.wait_all_tasks_blocked()
        assert not drained  # Blocked, as the buffer is not empty

        await stream_buffer.pop(20)
        await anyio.wait_all_tasks_blocked()
        assert drained


@pytest.mark.anyio
async def test_stream_buffer_closed() -> None:
    stream_buffer = StreamBuffer(EventWrapper)
    await stream_buffer.close()
    await stream_buffer._is_empty.wait()
    await stream_buffer._paused.wait()
    assert True
    with pytest.raises(BufferCompleteError):
        await stream_buffer.push(b"a")


@pytest.mark.anyio
async def test_stream_buffer_complete() -> None:
    stream_buffer = StreamBuffer(EventWrapper)
    await stream_buffer.push(b"a" * 10)
    assert not stream_buffer.complete
    stream_buffer.set_complete()
    assert not stream_buffer.complete
    await stream_buffer.pop(20)
    assert stream_buffer.complete


@pytest.mark.anyio
async def test_protocol_handle_protocol_error() -> None:
    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
    )
    await protocol.handle(RawData(data=b"broken nonsense\r\n\r\n"))
    protocol.send.assert_awaited()  # type: ignore[attr-defined]
    assert protocol.send.call_args_list == [call(Closed())]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_protocol_keep_alive_max_requests() -> None:
    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
    )
    protocol.config.keep_alive_max_requests = 0
    client = H2Connection()
    client.initiate_connection()
    headers = [
        (":method", "GET"),
        (":path", "/reqinfo"),
        (":authority", "anycorn"),
        (":scheme", "https"),
    ]
    client.send_headers(1, headers, end_stream=True)
    await protocol.handle(RawData(data=client.data_to_send()))
    protocol.send.assert_awaited()  # type: ignore[attr-defined]
    events = client.receive_data(protocol.send.call_args_list[1].args[0].data)  # type: ignore[attr-defined]
    assert isinstance(events[-1], ConnectionTerminated)


@pytest.mark.anyio
async def test_connect_without_path_does_not_crash() -> None:
    """An HTTP/2 CONNECT with no :path must be handled, not crash the connection.

    A CONNECT is routed to a WebSocket stream, and _create_stream read a :path that a
    CONNECT need not carry. Unlike aioquic over HTTP/3, the h2 library rejects such a
    request before it reaches _create_stream (no stream is created), which is why the
    default there is defensive - this pins that a non-conformant peer sending one is
    turned away cleanly rather than taking the connection down.
    """
    sent: list[object] = []

    async def send(event: object) -> None:
        sent.append(event)

    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        send,
        None,
    )
    # A conformant client will not send a CONNECT without :path, so outbound
    # validation is disabled to put one on the wire, as a hostile peer could.
    client = H2Connection(
        config=H2Configuration(
            client_side=True,
            validate_outbound_headers=False,
            normalize_outbound_headers=False,
        )
    )
    client.initiate_connection()
    client.send_headers(
        1,
        [(b":method", b"CONNECT"), (b":authority", b"anycorn:443")],
        end_stream=True,
    )

    # Must not raise, and no stream is handed to the app.
    await protocol.handle(RawData(data=client.data_to_send()))
    assert protocol.streams == {}


@pytest.mark.anyio
async def test_stream_send_trailers_ends_stream() -> None:
    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
    )
    protocol.connection.send_headers = Mock()  # type: ignore[method-assign]
    protocol.priority.insert_stream(1)
    protocol.stream_buffers[1] = StreamBuffer(EventWrapper)
    protocol.stream_buffers[1].set_complete()
    await protocol.stream_buffers[1]._is_empty.set()

    with anyio.fail_after(2):
        await protocol.stream_send(Trailers(stream_id=1, headers=[(b"x", b"y")]))

    protocol.connection.send_headers.assert_called_once_with(1, [(b"x", b"y")], end_stream=True)


async def _push_once_app(
    scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable
) -> None:
    """Respond to any request; on "/" also push one resource first."""
    assert scope["type"] == "http"
    while True:
        event = await receive()
        if not event.get("more_body", False):
            break
    if scope["path"] == "/":
        await send({"type": "http.response.push", "path": "/pushed", "headers": []})
    body = scope["path"].encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", b"%d" % len(body))],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


@pytest.mark.anyio
async def test_server_push_counts_once_towards_keep_alive() -> None:
    """A pushed stream must count once toward keep_alive_requests, like a request.

    _create_server_push builds the pushed stream through _create_stream, which already
    increments keep_alive_requests; a second increment counted every push twice, so a
    connection using server push hit keep_alive_max_requests and closed too early.

    Driven through a real task group and a real ASGI app that emits an
    http.response.push, so the count reflects what actually flows, not a stubbed path.
    """
    sent = bytearray()

    async def send(event: Event) -> None:
        if isinstance(event, RawData):
            sent.extend(event.data)

    async with TaskGroup() as task_group:
        protocol = H2Protocol(
            ASGIWrapper(_push_once_app),
            Config(),
            WorkerContext(None),
            task_group,
            ConnectionState({}),
            ("127.0.0.1", 80),
            ("127.0.0.1", 8000),
            send,
            None,
        )
        await protocol.initiate()

        client = H2Connection()
        client.initiate_connection()
        client.send_headers(
            1,
            [
                (b":method", b"GET"),
                (b":path", b"/"),
                (b":authority", b"anycorn"),
                (b":scheme", b"https"),
            ],
            end_stream=True,
        )
        await protocol.handle(RawData(data=client.data_to_send()))
        # Let the app run: receive the request, emit the push, and respond.
        await anyio.wait_all_tasks_blocked()

        # The push promise really went out (the else-branch ran, not the refusal)...
        events = client.receive_data(bytes(sent))
        assert any(isinstance(event, PushedStreamReceived) for event in events)
        # ... and it added exactly one: request (1) + pushed stream (1), not 3.
        assert protocol.keep_alive_requests == 2  # noqa: PLR2004

        await protocol.handle(Closed())  # stop the send task so the group can exit


@pytest.mark.anyio
async def test_reset_stream_does_not_leak_send_state() -> None:
    """A reset stream must not leave its send buffer and priority entry behind.

    Only _send_data tidies those up, and only once a response has finished. A
    stream the peer resets never gets there, so on a long-lived connection - a
    browser cancelling requests, say - both grew without bound.
    """
    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
    )
    client = H2Connection()
    client.initiate_connection()
    client.send_headers(
        1,
        [
            (b":method", b"POST"),
            (b":path", b"/"),
            (b":authority", b"anycorn"),
            (b":scheme", b"https"),
        ],
        end_stream=False,
    )
    await protocol.handle(RawData(data=client.data_to_send()))
    assert 1 in protocol.stream_buffers
    assert 1 in protocol.priority._streams

    client.reset_stream(1)
    await protocol.handle(RawData(data=client.data_to_send()))

    assert protocol.streams == {}
    assert protocol.stream_buffers == {}
    assert 1 not in protocol.priority._streams


@pytest.mark.anyio
async def test_idle_reflects_open_streams() -> None:
    """Idle is true with no streams, and follows the streams' own idle state otherwise."""
    protocol = _protocol()
    assert protocol.idle
    protocol.streams[1] = Mock(idle=False)
    assert not protocol.idle
    protocol.streams[1] = Mock(idle=True)
    assert protocol.idle


@pytest.mark.anyio
async def test_handle_ignores_an_unknown_event() -> None:
    """Handle only reacts to RawData and Closed; anything else is a no-op."""
    protocol = _protocol()
    await protocol.handle(Updated(idle=True))
    protocol.send.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_handle_closed_closes_open_streams() -> None:
    """Closing the connection closes each open stream and stops the send task."""
    protocol = _protocol()
    stream = AsyncMock()
    protocol.streams[1] = stream
    await protocol.handle(Closed())
    assert protocol.closed
    stream.handle.assert_awaited()  # a StreamClosed reached the stream


@pytest.mark.anyio
async def test_close_stream_that_is_gone_is_a_noop() -> None:
    """Closing a stream that is not open must not fail."""
    protocol = _protocol()
    await protocol._close_stream(999)


@pytest.mark.anyio
async def test_discard_send_state_without_a_buffer() -> None:
    """Discarding send state for a stream with no buffer is safe and idempotent."""
    protocol = _protocol()
    await protocol._discard_send_state(999)


@pytest.mark.anyio
async def test_send_data_discards_state_when_the_buffer_is_gone() -> None:
    """A stream in the priority tree but without a buffer is forced closed, not crashed."""
    stream_id = 5
    protocol = _protocol()
    protocol.priority.insert_stream(stream_id)
    await protocol._send_data(stream_id)  # KeyError on the missing buffer -> discard
    assert stream_id not in protocol.priority._streams


@pytest.mark.anyio
async def test_stream_send_swallows_a_missing_stream() -> None:
    """A stream event for a stream that has gone is dropped rather than raising."""
    protocol = _protocol()
    await protocol.stream_send(Body(stream_id=7, data=b"x"))  # no buffer/priority entry


@pytest.mark.anyio
async def test_stream_send_ignores_an_unhandled_event() -> None:
    """A stream event h2 does not act on falls through without error."""
    protocol = _protocol()
    await protocol.stream_send(StreamEvent(stream_id=1))


@pytest.mark.anyio
async def test_a_priority_frame_reprioritizes_a_known_stream() -> None:
    """A PRIORITY frame for an existing stream reprioritizes it in the tree."""
    protocol = _protocol()
    client = H2Connection()
    client.initiate_connection()
    await protocol.handle(RawData(data=_client_request(client)))
    client.prioritize(1, weight=200, depends_on=0, exclusive=False)
    await protocol.handle(RawData(data=client.data_to_send()))  # PriorityUpdated -> reprioritize


@pytest.mark.anyio
async def test_a_priority_frame_before_headers_is_kept_and_then_reused() -> None:
    """PRIORITY before HEADERS inserts the stream; the later HEADERS reuses that entry."""
    protocol = _protocol()
    client = H2Connection()
    client.initiate_connection()
    await protocol.handle(RawData(data=client.data_to_send()))
    client.prioritize(1, weight=50, depends_on=0, exclusive=False)  # before any HEADERS
    await protocol.handle(RawData(data=client.data_to_send()))  # MissingStreamError -> insert
    await protocol.handle(RawData(data=_client_request(client)))  # DuplicateStreamError -> kept
    assert 1 in protocol.streams


@pytest.mark.anyio
async def test_a_settings_change_without_a_window_size_is_handled() -> None:
    """A remote SETTINGS change that does not touch the window size is applied cleanly."""
    protocol = _protocol()
    client = H2Connection()
    client.initiate_connection()
    await protocol.handle(RawData(data=client.data_to_send()))
    client.update_settings({SettingCodes.HEADER_TABLE_SIZE: 4096})
    await protocol.handle(RawData(data=client.data_to_send()))


@pytest.mark.anyio
async def test_a_goaway_closes_the_connection() -> None:
    """A GOAWAY from the peer surfaces as a Closed to the connection."""
    sent: list[object] = []

    async def send(event: object) -> None:
        sent.append(event)

    protocol = _protocol(send=send)
    client = H2Connection()
    client.initiate_connection()
    await protocol.handle(RawData(data=client.data_to_send()))
    client.close_connection()  # GOAWAY
    await protocol.handle(RawData(data=client.data_to_send()))
    assert Closed() in sent


@pytest.mark.anyio
async def test_a_refused_push_is_swallowed() -> None:
    """When the client has disabled push, the app's push attempt is dropped, not fatal."""
    sent = bytearray()

    async def send(event: Event) -> None:
        if isinstance(event, RawData):
            sent.extend(event.data)

    async with TaskGroup() as task_group:
        protocol = H2Protocol(
            ASGIWrapper(_push_once_app),
            Config(),
            WorkerContext(None),
            task_group,
            ConnectionState({}),
            ("127.0.0.1", 80),
            ("127.0.0.1", 8000),
            send,
            None,
        )
        await protocol.initiate()
        client = H2Connection()
        client.initiate_connection()
        client.update_settings({SettingCodes.ENABLE_PUSH: 0})  # refuse pushes
        await protocol.handle(RawData(data=client.data_to_send()))
        # A body in two parts, so the app's receive loop drains more than one message.
        client.send_headers(
            1,
            [
                (b":method", b"POST"),
                (b":path", b"/"),
                (b":authority", b"a"),
                (b":scheme", b"https"),
            ],
            end_stream=False,
        )
        client.send_data(1, b"body", end_stream=True)
        await protocol.handle(RawData(data=client.data_to_send()))
        await anyio.wait_all_tasks_blocked()
        # No push promise was accepted, and only the one request was counted.
        events = client.receive_data(bytes(sent))
        assert not any(isinstance(event, PushedStreamReceived) for event in events)
        assert protocol.keep_alive_requests == 1
        await protocol.handle(Closed())
