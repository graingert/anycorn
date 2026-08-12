"""Tests for WebSocket stream implementation."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, call

import anyio
import anyio.lowlevel
import pytest
import wsproto.connection
import wsproto.events
from wsproto.connection import ConnectionState as WSConnectionState
from wsproto.connection import ConnectionType
from wsproto.events import BytesMessage, CloseConnection, Ping, TextMessage

from anycorn.config import Config
from anycorn.protocol.events import Body, Data, EndBody, EndData, Request, Response, StreamClosed
from anycorn.protocol.ws_stream import (
    ASGIWebsocketState,
    FrameTooLargeError,
    Handshake,
    WebsocketBuffer,
    WSStream,
)
from anycorn.task_group import TaskGroup
from anycorn.typing import (
    ConnectionState,
    WebsocketAcceptEvent,
    WebsocketCloseEvent,
    WebsocketResponseBodyEvent,
    WebsocketResponseStartEvent,
    WebsocketSendEvent,
)
from anycorn.utils import UnexpectedMessageError, default_tls_extension
from anycorn.worker_context import WorkerContext
from tests.helpers import LogCapture, capture_logs

# WebSocket frame opcodes (first byte, FIN set), for asserting the kind of frame sent.
_BINARY_FRAME = 0x82
_CLOSE_FRAME = 0x88
_PONG_FRAME = 0x8A


def test_buffer() -> None:
    buffer_ = WebsocketBuffer(10)
    buffer_.extend(TextMessage(data="abc", frame_finished=False, message_finished=True))
    assert buffer_.to_message() == {"type": "websocket.receive", "bytes": None, "text": "abc"}
    buffer_.clear()
    buffer_.extend(BytesMessage(data=b"abc", frame_finished=False, message_finished=True))
    assert buffer_.to_message() == {"type": "websocket.receive", "bytes": b"abc", "text": None}


def test_buffer_frame_too_large() -> None:
    buffer_ = WebsocketBuffer(2)
    with pytest.raises(FrameTooLargeError):
        buffer_.extend(TextMessage(data="abc", frame_finished=False, message_finished=True))


@pytest.mark.parametrize(
    "data",
    [
        (
            TextMessage(data="abc", frame_finished=False, message_finished=True),
            BytesMessage(data=b"abc", frame_finished=False, message_finished=True),
        ),
        (
            BytesMessage(data=b"abc", frame_finished=False, message_finished=True),
            TextMessage(data="abc", frame_finished=False, message_finished=True),
        ),
    ],
)
def test_buffer_mixed_types(data: list) -> None:
    buffer_ = WebsocketBuffer(10)
    buffer_.extend(data[0])
    with pytest.raises(TypeError):
        buffer_.extend(data[1])


@pytest.mark.parametrize(
    ("headers", "http_version", "valid"),
    [
        ([], "1.0", False),
        (
            [
                (b"connection", b"upgrade, keep-alive"),
                (b"sec-websocket-version", b"13"),
                (b"upgrade", b"websocket"),
                (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
            ],
            "1.1",
            True,
        ),
        (
            [
                (b"connection", b"keep-alive"),
                (b"sec-websocket-version", b"13"),
                (b"upgrade", b"websocket"),
                (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
            ],
            "1.1",
            False,
        ),
        (
            [
                (b"connection", b"upgrade, keep-alive"),
                (b"sec-websocket-version", b"13"),
                (b"upgrade", b"h2c"),
                (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
            ],
            "1.1",
            False,
        ),
        ([(b"sec-websocket-version", b"13")], "2", True),
        ([(b"sec-websocket-version", b"12")], "2", False),
    ],
)
def test_handshake_validity(
    headers: list[tuple[bytes, bytes]],
    http_version: str,
    valid: bool,  # noqa: FBT001
) -> None:
    handshake = Handshake(headers, http_version)
    assert handshake.is_valid() is valid


def test_handshake_accept_http1() -> None:
    handshake = Handshake(
        [
            (b"connection", b"upgrade, keep-alive"),
            (b"sec-websocket-version", b"13"),
            (b"upgrade", b"websocket"),
            (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
        ],
        "1.1",
    )
    status_code, headers, _ = handshake.accept(None, [])
    assert status_code == 101  # noqa: PLR2004
    assert headers == [
        (b"sec-websocket-accept", b"1BpNk/3ah1huDGgcuMJBcjcMbEA="),
        (b"upgrade", b"WebSocket"),
        (b"connection", b"Upgrade"),
    ]


def test_handshake_accept_http2() -> None:
    handshake = Handshake([(b"sec-websocket-version", b"13")], "2")
    status_code, headers, _ = handshake.accept(None, [])
    assert status_code == 200  # noqa: PLR2004
    assert headers == []


def test_handshake_accept_additional_headers() -> None:
    handshake = Handshake(
        [
            (b"connection", b"upgrade, keep-alive"),
            (b"sec-websocket-version", b"13"),
            (b"upgrade", b"websocket"),
            (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
        ],
        "1.1",
    )
    status_code, headers, _ = handshake.accept(None, [(b"additional", b"header")])
    assert status_code == 101  # noqa: PLR2004
    assert headers == [
        (b"sec-websocket-accept", b"1BpNk/3ah1huDGgcuMJBcjcMbEA="),
        (b"upgrade", b"WebSocket"),
        (b"connection", b"Upgrade"),
        (b"additional", b"header"),
    ]


@pytest.fixture(name="config")
def _config() -> Config:
    return Config()


@pytest.fixture(name="logs")
def _logs(config: Config) -> LogCapture:
    """Run the real access log against *config*, and collect what it writes."""
    return capture_logs(config)


@pytest.fixture(name="stream")
async def _stream(config: Config, logs: LogCapture) -> WSStream:  # noqa: ARG001
    stream = WSStream(
        AsyncMock(),
        config,
        WorkerContext(None),
        AsyncMock(),
        None,
        None,
        AsyncMock(),
        1,
        None,
    )
    stream.task_group.spawn_app.return_value = AsyncMock()  # type: ignore[attr-defined]
    stream.app_put = AsyncMock()
    return stream


@pytest.mark.anyio
async def test_handle_request(stream: WSStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    stream.task_group.spawn_app.assert_called()  # type: ignore[attr-defined]
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    assert scope == {
        "type": "websocket",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "scheme": "ws",
        "http_version": "2",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"a=b",
        "root_path": "",
        "headers": [(b"sec-websocket-version", b"13")],
        "client": None,
        "server": None,
        "subprotocols": [],
        "extensions": {"websocket.http.response": {}},
        "state": ConnectionState({}),
    }


@pytest.mark.anyio
async def test_handle_request_tls() -> None:
    stream = WSStream(
        AsyncMock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        None,
        None,
        AsyncMock(),
        1,
        default_tls_extension(),
    )
    stream.task_group.spawn_app.return_value = AsyncMock()  # type: ignore[attr-defined]
    stream.app_put = AsyncMock()
    capture_logs(stream.config)
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    assert "tls" in scope["extensions"]
    assert scope["extensions"]["tls"]["client_cert_chain"] == ()
    assert scope["scheme"] == "wss"


@pytest.mark.anyio
async def test_handle_data_before_acceptance(stream: WSStream) -> None:
    """Data arriving before the app accepts is held, not answered.

    Whether the app has accepted by the time these bytes are handled is down to
    the event loop - it had, under asyncio, and had not under trio, so the same
    client got a 200 or a 400 depending on the backend. Holding them takes the
    scheduler out of it; they are delivered once the handshake resolves.
    """
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.handle(Data(stream_id=1, data=b"X"))

    assert stream.send.call_args_list == []  # type: ignore[attr-defined]
    assert stream.pre_accept_data == b"X"
    assert not stream.closed


@pytest.mark.anyio
async def test_data_before_acceptance_is_delivered_once_accepted(stream: WSStream) -> None:
    """The held bytes reach the app as soon as there is a connection to decode them."""
    client = wsproto.connection.Connection(wsproto.ConnectionType.CLIENT)
    frame = client.send(wsproto.events.BytesMessage(data=b"early"))

    await stream.handle(
        Request(
            stream_id=1,
            http_version="1.1",
            headers=[
                (b"host", b"anycorn"),
                (b"connection", b"upgrade"),
                (b"upgrade", b"websocket"),
                (b"sec-websocket-key", b"UnQ3lpJAH6j2PslA993iKQ=="),
                (b"sec-websocket-version", b"13"),
            ],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.handle(Data(stream_id=1, data=frame))
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))

    assert stream.pre_accept_data == b""
    assert stream.app_put.call_args_list[-1] == call(  # type: ignore[attr-defined]
        {"type": "websocket.receive", "bytes": b"early", "text": None}
    )


@pytest.mark.anyio
async def test_data_before_acceptance_is_bounded(stream: WSStream) -> None:
    """A peer that keeps sending at an app slow to accept is cut off, not buffered."""
    stream.config.websocket_max_message_size = 4
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )

    await stream.handle(Data(stream_id=1, data=b"toolong"))

    assert stream.closed
    assert stream.send.call_args_list[0][0][0].status_code == HTTPStatus.BAD_REQUEST  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_handle_connection(stream: WSStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
    stream.app_put = AsyncMock()
    await stream.handle(Data(stream_id=1, data=b"\x81\x85&`\x13\x0eN\x05\x7fbI"))
    stream.app_put.assert_called()
    assert stream.app_put.call_args_list == [
        call({"type": "websocket.receive", "bytes": None, "text": "hello"})
    ]


@pytest.mark.anyio
async def test_handle_closed(stream: WSStream) -> None:
    await stream.handle(StreamClosed(stream_id=1))
    stream.app_put.assert_called()  # type: ignore[attr-defined]
    assert stream.app_put.call_args_list == [call({"type": "websocket.disconnect", "code": 1006})]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_handle_client_close_reports_its_code(stream: WSStream) -> None:
    """A clean client close must reach the app as its own code, not 1006.

    Before, the CloseConnection from the peer was not recorded, so when the stream then
    closed the disconnect defaulted to ABNORMAL_CLOSURE - the app could not tell a
    graceful client close from a dropped connection.

    https://github.com/pgjones/hypercorn/issues/127
    """
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
    stream.app_put = AsyncMock()

    # A masked client close frame carrying code 1000 (a zero mask key leaves the
    # 2-byte payload - 0x03e8 - unchanged).
    await stream.handle(Data(stream_id=1, data=b"\x88\x82\x00\x00\x00\x00\x03\xe8"))
    await stream.handle(StreamClosed(stream_id=1))

    assert stream.app_put.call_args_list == [call({"type": "websocket.disconnect", "code": 1000})]


@pytest.mark.anyio
async def test_send_accept(stream: WSStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
    assert stream.state == ASGIWebsocketState.CONNECTED
    stream.send.assert_called()  # type: ignore[attr-defined]
    assert stream.send.call_args_list == [call(Response(stream_id=1, headers=[], status_code=200))]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_send_accept_with_additional_headers(stream: WSStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(
        cast(
            "WebsocketAcceptEvent",
            {"type": "websocket.accept", "headers": [(b"additional", b"header")]},
        )
    )
    assert stream.state == ASGIWebsocketState.CONNECTED
    stream.send.assert_called()  # type: ignore[attr-defined]
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(Response(stream_id=1, headers=[(b"additional", b"header")], status_code=200))
    ]


@pytest.mark.anyio
async def test_send_reject(stream: WSStream, logs: LogCapture) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(
        cast(
            "WebsocketResponseStartEvent",
            {"type": "websocket.http.response.start", "status": 200, "headers": []},
        ),
    )
    assert stream.state == ASGIWebsocketState.HANDSHAKE
    # Must wait for response before sending anything
    stream.send.assert_not_called()  # type: ignore[attr-defined]
    await stream.app_send(
        cast(
            "WebsocketResponseBodyEvent",
            {"type": "websocket.http.response.body", "body": b"Body"},
        )
    )
    assert stream.state == ASGIWebsocketState.HTTPCLOSED
    stream.send.assert_called()  # type: ignore[unresolved-attribute]
    assert stream.send.call_args_list == [  # type: ignore[unresolved-attribute]
        call(Response(stream_id=1, headers=[], status_code=200)),
        call(Body(stream_id=1, data=b"Body")),
        call(EndBody(stream_id=1)),
    ]
    assert len(logs.access) == 1


@pytest.mark.anyio
async def test_invalid_server_name(stream: WSStream) -> None:
    stream.config.server_names = ["anycorn"]
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"host", b"example.com"), (b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(
            Response(
                stream_id=1,
                headers=[(b"content-length", b"0"), (b"connection", b"close")],
                status_code=404,
            )
        ),
        call(EndBody(stream_id=1)),
        call(StreamClosed(stream_id=1)),
    ]
    # This shouldn't error
    await stream.handle(Body(stream_id=1, data=b"Body"))


@pytest.mark.anyio
async def test_send_app_error_handshake(stream: WSStream, logs: LogCapture) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(None)
    stream.send.assert_called()  # type: ignore[attr-defined]
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(
            Response(
                stream_id=1,
                headers=[(b"content-length", b"0"), (b"connection", b"close")],
                status_code=500,
            )
        ),
        call(EndBody(stream_id=1)),
        call(StreamClosed(stream_id=1)),
    ]
    assert len(logs.access) == 1


@pytest.mark.anyio
async def test_send_app_error_connected(stream: WSStream, logs: LogCapture) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
    await stream.app_send(None)
    stream.send.assert_called()  # type: ignore[attr-defined]
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(Response(stream_id=1, headers=[], status_code=200)),
        call(Data(stream_id=1, data=b"\x88\x02\x03\xf3")),
        call(StreamClosed(stream_id=1)),
    ]
    assert len(logs.access) == 1


@pytest.mark.anyio
async def test_send_connection(stream: WSStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
    await stream.app_send(cast("WebsocketSendEvent", {"type": "websocket.send", "text": "hello"}))
    await stream.app_send(cast("WebsocketCloseEvent", {"type": "websocket.close"}))
    stream.send.assert_called()  # type: ignore[attr-defined]
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(Response(stream_id=1, headers=[], status_code=200)),
        call(Data(stream_id=1, data=b"\x81\x05hello")),
        call(Data(stream_id=1, data=b"\x88\x02\x03\xe8")),
        call(EndData(stream_id=1)),
    ]


class _PingClock:
    """Stands in for the ping loop's sleep, so the test decides how often it wakes.

    Lets a fixed number of intervals elapse instantly and then parks, which pins the
    number of pings to exactly what the test asked for. Sleeping for real instead
    would tie the count to how fast the machine happens to be.
    """

    def __init__(self, intervals: int) -> None:
        self.waits: list[float] = []
        self._remaining = intervals
        self._released = anyio.Event()

    async def sleep(self, wait: float) -> None:
        self.waits.append(wait)
        if self._remaining > 0:
            self._remaining -= 1
            await anyio.lowlevel.checkpoint()
            return
        await self._released.wait()

    def release(self) -> None:
        """Let the parked sleep return, so the loop can notice the stream closed."""
        self._released.set()


@pytest.mark.anyio
async def test_pings(stream: WSStream) -> None:
    stream.config.websocket_ping_interval = 0.1
    clock = _PingClock(intervals=1)
    stream.context.sleep = clock.sleep  # type: ignore[method-assign]
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"sec-websocket-version", b"13")],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    async with TaskGroup() as task_group:
        stream.task_group = task_group
        await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
        stream.app_put = AsyncMock()
        # One interval elapses, so the loop pings, waits, pings, and then parks
        await anyio.wait_all_tasks_blocked()
        assert stream.send.call_args_list == [  # type: ignore[attr-defined]
            call(Response(stream_id=1, headers=[], status_code=200)),
            call(Data(stream_id=1, data=b"\x89\x00")),
            call(Data(stream_id=1, data=b"\x89\x00")),
        ]
        assert clock.waits == [0.1, 0.1]
        await stream.handle(StreamClosed(stream_id=1))
        clock.release()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("state", "message_type"),
    [
        (ASGIWebsocketState.HANDSHAKE, "websocket.send"),
        (ASGIWebsocketState.RESPONSE, "websocket.accept"),
        (ASGIWebsocketState.RESPONSE, "websocket.send"),
        (ASGIWebsocketState.CONNECTED, "websocket.http.response.start"),
        (ASGIWebsocketState.CONNECTED, "websocket.http.response.body"),
        (ASGIWebsocketState.CLOSED, "websocket.send"),
        (ASGIWebsocketState.CLOSED, "websocket.http.response.start"),
        (ASGIWebsocketState.CLOSED, "websocket.http.response.body"),
    ],
)
async def test_send_invalid_message_given_state(
    stream: WSStream, state: ASGIWebsocketState, message_type: str
) -> None:
    stream.state = state
    with pytest.raises(UnexpectedMessageError):
        await stream.app_send({"type": message_type})  # type: ignore[arg-type]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "headers", "body"),
    [
        ("201 NO CONTENT", [], b""),  # Status should be int
        (200, [("X-Foo", "foo")], b""),  # Headers should be bytes
        (200, [], "Body"),  # Body should be bytes
    ],
)
async def test_send_invalid_http_message(
    stream: WSStream,
    status: Any,  # noqa: ANN401
    headers: Any,  # noqa: ANN401
    body: Any,  # noqa: ANN401
) -> None:
    stream.connection = Mock()
    stream.state = ASGIWebsocketState.HANDSHAKE
    stream.scope = {"method": "GET"}  # type: ignore[typeddict-item, typeddict-unknown-key]
    with pytest.raises((TypeError, ValueError)):  # noqa: PT012
        await stream.app_send(
            cast(
                "WebsocketResponseStartEvent",
                {"type": "websocket.http.response.start", "headers": headers, "status": status},
            ),
        )
        await stream.app_send(
            cast(
                "WebsocketResponseBodyEvent",
                {"type": "websocket.http.response.body", "body": body},
            )
        )


@pytest.mark.parametrize(
    ("state", "idle"),
    [
        (state, False)
        for state in ASGIWebsocketState
        if state not in {ASGIWebsocketState.CLOSED, ASGIWebsocketState.HTTPCLOSED}
    ]
    + [(ASGIWebsocketState.CLOSED, True), (ASGIWebsocketState.HTTPCLOSED, True)],
)
@pytest.mark.anyio
async def test_stream_idle(stream: WSStream, state: ASGIWebsocketState, idle: bool) -> None:  # noqa: FBT001
    stream.state = state
    assert stream.idle is idle


@pytest.mark.anyio
async def test_closure(stream: WSStream) -> None:
    assert not stream.closed
    await stream.handle(StreamClosed(stream_id=1))
    assert stream.closed
    await stream.handle(StreamClosed(stream_id=1))
    assert stream.closed
    # It is important that the disconnect message has only been sent
    # once.
    assert stream.app_put.call_args_list == [call({"type": "websocket.disconnect", "code": 1006})]  # type: ignore[unresolved-attribute]


@pytest.mark.anyio
async def test_closed_app_send_noop(stream: WSStream) -> None:
    stream.closed = True
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))
    stream.send.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_rejection_body_without_a_response_start(stream: WSStream) -> None:
    """A denial body with no response started is the app's error, not a crash.

    The status to send it with comes from the response start that never arrived,
    and reading it raised AttributeError - so an app that sent the body first got
    that instead of the unexpected-message error describing what it did wrong.
    """
    await stream.handle(
        Request(
            stream_id=1,
            http_version="1.1",
            method="GET",
            raw_path=b"/",
            state=ConnectionState({}),
            headers=[
                (b"host", b"anycorn"),
                (b"connection", b"Upgrade"),
                (b"upgrade", b"WebSocket"),
                (b"sec-websocket-version", b"13"),
                (b"sec-websocket-key", b"ZGVtbw=="),
            ],
        )
    )

    with pytest.raises(UnexpectedMessageError):
        await stream.app_send(
            {"type": "websocket.http.response.body", "body": b"denied", "more_body": False}
        )


def test_handshake_parses_extensions_and_subprotocols() -> None:
    """The extension and subprotocol headers are split into lists."""
    handshake = Handshake(
        [
            (b"sec-websocket-extensions", b"permessage-deflate"),
            (b"sec-websocket-protocol", b"chat, superchat"),
        ],
        "2",
    )
    assert handshake.extensions == ["permessage-deflate"]
    assert handshake.subprotocols == ["chat", "superchat"]


def test_handshake_http1_without_a_key_is_invalid() -> None:
    """An HTTP/1.1 upgrade with no Sec-WebSocket-Key cannot be accepted."""
    handshake = Handshake(
        [
            (b"connection", b"upgrade"),
            (b"upgrade", b"websocket"),
            (b"sec-websocket-version", b"13"),
        ],
        "1.1",
    )
    assert handshake.is_valid() is False


def test_handshake_accept_with_a_valid_subprotocol() -> None:
    """A subprotocol the client offered is echoed back in the accept headers."""
    handshake = Handshake(
        [(b"sec-websocket-version", b"13"), (b"sec-websocket-protocol", b"chat")], "2"
    )
    _, headers, _ = handshake.accept("chat", [])
    assert (b"sec-websocket-protocol", b"chat") in headers


def test_handshake_accept_with_an_unoffered_subprotocol_raises() -> None:
    """A subprotocol the client never offered cannot be selected."""
    handshake = Handshake([(b"sec-websocket-version", b"13")], "2")
    with pytest.raises(Exception, match="Invalid Subprotocol"):
        handshake.accept("nope", [])


def test_handshake_accept_negotiates_permessage_deflate() -> None:
    """When the client offers permessage-deflate and it is enabled, it is accepted."""
    handshake = Handshake(
        [(b"sec-websocket-version", b"13"), (b"sec-websocket-extensions", b"permessage-deflate")],
        "2",
        websocket_permessage_deflate=True,
    )
    _, headers, _ = handshake.accept(None, [])
    assert any(name == b"sec-websocket-extensions" for name, _ in headers)


@pytest.mark.parametrize("bad", [(b"sec-websocket-protocol", b"x"), (b":method", b"GET")])
def test_handshake_accept_rejects_reserved_additional_headers(bad: tuple[bytes, bytes]) -> None:
    """The app cannot smuggle a subprotocol or pseudo-header through additional_headers."""
    handshake = Handshake([(b"sec-websocket-version", b"13")], "2")
    with pytest.raises(Exception, match="Invalid additional header"):
        handshake.accept(None, [bad])


def _ws_request(http_version: str = "2", *, headers: Any = None, raw_path: bytes = b"/") -> Request:  # noqa: ANN401
    return Request(
        stream_id=1,
        http_version=http_version,
        headers=headers if headers is not None else [(b"sec-websocket-version", b"13")],
        raw_path=raw_path,
        method="GET",
        state=ConnectionState({}),
    )


async def _accept(stream: WSStream) -> None:
    await stream.handle(_ws_request())
    await stream.app_send(cast("WebsocketAcceptEvent", {"type": "websocket.accept"}))


@pytest.mark.anyio
async def test_handle_rejects_an_invalid_path(stream: WSStream) -> None:
    """A malformed percent-encoded path is answered with a 400 before the app runs."""
    await stream.handle(_ws_request(raw_path=b"/%zz"))
    assert stream.closed
    assert stream.send.call_args_list[0][0][0].status_code == HTTPStatus.BAD_REQUEST  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_handle_rejects_an_invalid_handshake(stream: WSStream) -> None:
    """An HTTP/1.1 request missing the WebSocket key is rejected with 400."""
    await stream.handle(_ws_request("1.1", headers=[(b"host", b"anycorn")]))
    assert stream.closed
    assert stream.send.call_args_list[0][0][0].status_code == HTTPStatus.BAD_REQUEST  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_handle_ignores_an_unexpected_event(stream: WSStream) -> None:
    """Handle only reacts to Request, Body/Data and StreamClosed; other events are no-ops."""
    await stream.handle(EndBody(stream_id=1))
    stream.send.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_stream_closed_without_an_app_is_a_noop(stream: WSStream) -> None:
    """StreamClosed before the app was spawned closes without a disconnect message."""
    stream.app_put = None
    await stream.handle(StreamClosed(stream_id=1))
    assert stream.closed


@pytest.mark.anyio
async def test_stream_closed_after_a_clean_close_reports_normal_closure(stream: WSStream) -> None:
    """Closing from a CLOSED state reports a normal closure, not 1006."""
    stream.state = ASGIWebsocketState.CLOSED
    await stream.handle(StreamClosed(stream_id=1))
    assert stream.app_put.call_args_list == [  # type: ignore[attr-defined]
        call({"type": "websocket.disconnect", "code": 1000})
    ]


@pytest.mark.anyio
async def test_send_a_bytes_message(stream: WSStream) -> None:
    """A websocket.send carrying bytes frames a binary message."""
    await _accept(stream)
    stream.send.reset_mock()  # type: ignore[attr-defined]
    await stream.app_send(
        cast("WebsocketSendEvent", {"type": "websocket.send", "bytes": b"hi", "text": None})
    )
    data = stream.send.call_args_list[0][0][0]  # type: ignore[attr-defined]
    assert isinstance(data, Data)
    assert data.data[0] == _BINARY_FRAME


@pytest.mark.anyio
async def test_send_with_non_str_text_raises(stream: WSStream) -> None:
    """websocket.send with a non-str text and no bytes is a type error."""
    await _accept(stream)
    with pytest.raises(TypeError, match="should be a str"):
        await stream.app_send(
            cast("WebsocketSendEvent", {"type": "websocket.send", "bytes": None, "text": 123})
        )


@pytest.mark.anyio
async def test_close_during_handshake_sends_403(stream: WSStream) -> None:
    """websocket.close before acceptance denies the upgrade with a 403."""
    await stream.handle(_ws_request())
    await stream.app_send(cast("WebsocketCloseEvent", {"type": "websocket.close"}))
    assert stream.state == ASGIWebsocketState.HTTPCLOSED
    assert stream.send.call_args_list[0][0][0].status_code == HTTPStatus.FORBIDDEN  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_app_finishing_in_response_state_just_closes(stream: WSStream) -> None:
    """An app that ends after starting a denial response closes without a wsproto close."""
    stream.state = ASGIWebsocketState.RESPONSE
    await stream.app_send(None)
    assert stream.send.call_args_list == [call(StreamClosed(stream_id=1))]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_send_wsproto_event_swallows_a_local_protocol_error(stream: WSStream) -> None:
    """Sending a message the wsproto connection will not allow is dropped, not raised."""
    connection = wsproto.connection.Connection(ConnectionType.SERVER)
    connection.send(CloseConnection(code=1000))  # now closing, so a message is rejected
    stream.connection = connection
    await stream._send_wsproto_event(TextMessage(data="late"))  # must not raise
    stream.send.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_rejection_body_streamed_over_several_messages(stream: WSStream) -> None:
    """A denial body sent in parts starts the response once and closes only at the end."""
    sent: list[Any] = []
    stream.send = AsyncMock(side_effect=sent.append)
    await stream.handle(_ws_request())
    await stream.app_send(
        cast(
            "WebsocketResponseStartEvent",
            {"type": "websocket.http.response.start", "status": 200, "headers": []},
        )
    )
    await stream.app_send(
        cast(
            "WebsocketResponseBodyEvent",
            {"type": "websocket.http.response.body", "body": b"part1", "more_body": True},
        )
    )
    assert stream.state == ASGIWebsocketState.RESPONSE  # started, not yet closed
    await stream.app_send(
        cast(
            "WebsocketResponseBodyEvent",
            {"type": "websocket.http.response.body", "body": b"part2", "more_body": False},
        )
    )
    assert stream.state == ASGIWebsocketState.HTTPCLOSED
    # Exactly one Response start, both body parts, one EndBody.
    assert [type(event).__name__ for event in sent] == ["Response", "Body", "Body", "EndBody"]


@pytest.mark.anyio
async def test_rejection_with_a_bodyless_status_sends_no_body(stream: WSStream) -> None:
    """A 204 denial suppresses the body but still finishes the response."""
    sent: list[Any] = []
    stream.send = AsyncMock(side_effect=sent.append)
    await stream.handle(_ws_request())
    await stream.app_send(
        cast(
            "WebsocketResponseStartEvent",
            {"type": "websocket.http.response.start", "status": 204, "headers": []},
        )
    )
    await stream.app_send(
        cast(
            "WebsocketResponseBodyEvent",
            {"type": "websocket.http.response.body", "body": b"ignored", "more_body": False},
        )
    )
    assert not any(isinstance(event, Body) for event in sent)
    assert any(isinstance(event, EndBody) for event in sent)


@pytest.mark.anyio
async def test_a_ping_is_answered_with_a_pong(stream: WSStream) -> None:
    """A ping from the peer is answered with a pong."""
    await _accept(stream)
    stream.send.reset_mock()  # type: ignore[attr-defined]
    client = wsproto.connection.Connection(ConnectionType.CLIENT)
    await stream.handle(Data(stream_id=1, data=client.send(Ping())))
    pong = stream.send.call_args_list[0][0][0]  # type: ignore[attr-defined]
    assert isinstance(pong, Data)
    assert pong.data[0] == _PONG_FRAME


@pytest.mark.anyio
async def test_a_pong_from_the_peer_is_ignored(stream: WSStream) -> None:
    """An unsolicited pong is neither answered nor delivered to the app."""
    await _accept(stream)
    stream.send.reset_mock()  # type: ignore[attr-defined]
    stream.app_put = AsyncMock()
    client = wsproto.connection.Connection(ConnectionType.CLIENT)
    await stream.handle(Data(stream_id=1, data=client.send(wsproto.events.Pong())))
    stream.send.assert_not_called()  # type: ignore[attr-defined]
    stream.app_put.assert_not_called()


@pytest.mark.anyio
async def test_a_message_over_the_limit_closes_with_message_too_big(stream: WSStream) -> None:
    """A message beyond the configured size closes the connection rather than buffering it."""
    await _accept(stream)
    stream.buffer.max_length = 4  # tighten after acceptance so the send below overflows
    stream.send.reset_mock()  # type: ignore[attr-defined]
    client = wsproto.connection.Connection(ConnectionType.CLIENT)
    await stream.handle(Data(stream_id=1, data=client.send(BytesMessage(data=b"far too long"))))
    close = stream.send.call_args_list[0][0][0]  # type: ignore[attr-defined]
    assert isinstance(close, Data)
    assert close.data[0] == _CLOSE_FRAME


@pytest.mark.anyio
async def test_a_fragmented_message_is_delivered_once_complete(stream: WSStream) -> None:
    """A message split across frames reaches the app only when the final frame lands."""
    await _accept(stream)
    stream.app_put = AsyncMock()
    client = wsproto.connection.Connection(ConnectionType.CLIENT)
    first = client.send(wsproto.events.Message(data="ab", message_finished=False))
    second = client.send(wsproto.events.Message(data="cd", message_finished=True))
    await stream.handle(Data(stream_id=1, data=first))
    stream.app_put.assert_not_called()  # nothing delivered mid-message
    await stream.handle(Data(stream_id=1, data=second))
    assert stream.app_put.call_args_list == [
        call({"type": "websocket.receive", "bytes": None, "text": "abcd"})
    ]


@pytest.mark.anyio
async def test_a_close_ack_after_we_closed_only_ends_the_stream(stream: WSStream) -> None:
    """Once the server has closed, the peer's close reply just ends the stream."""
    await _accept(stream)
    await stream.app_send(cast("WebsocketCloseEvent", {"type": "websocket.close"}))
    assert stream.connection.state is WSConnectionState.LOCAL_CLOSING
    stream.send.reset_mock()  # type: ignore[attr-defined]

    client = wsproto.connection.Connection(ConnectionType.CLIENT)
    client.receive_data(None)  # allow the client to reply to a close it will be told about
    await stream.handle(Data(stream_id=1, data=b"\x88\x82\x00\x00\x00\x00\x03\xe8"))
    # No close response is sent (we already closed); only the stream is ended.
    assert stream.send.call_args_list == [call(StreamClosed(stream_id=1))]  # type: ignore[attr-defined]
