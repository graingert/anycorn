"""Tests for HTTP stream implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, call

import pytest

from anycorn.config import Config
from anycorn.protocol.events import (
    Body,
    EndBody,
    Event,
    InformationalResponse,
    Request,
    Response,
    StreamClosed,
    Trailers,
    ZeroCopySend,
)
from anycorn.protocol.http_stream import ASGIHTTPState, HTTPStream
from anycorn.sendfile import have_sendfile
from anycorn.statsd import StatsdLogger
from anycorn.typing import (
    ConnectionState,
    HTTPResponseBodyEvent,
    HTTPResponsePathSendEvent,
    HTTPResponseStartEvent,
    HTTPScope,
)
from anycorn.utils import UnexpectedMessageError, default_tls_extension
from anycorn.worker_context import WorkerContext
from tests.helpers import LogCapture, capture_logs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(name="config")
def _config() -> Config:
    return Config()


@pytest.fixture(name="logs")
def _logs(config: Config) -> LogCapture:
    """Run the real access log against *config*, and collect what it writes."""
    return capture_logs(config)


@pytest.fixture(name="stream")
async def _stream(config: Config, logs: LogCapture) -> HTTPStream:  # noqa: ARG001
    stream = HTTPStream(
        AsyncMock(),
        config,
        WorkerContext(None),
        AsyncMock(),
        None,
        None,
        AsyncMock(),
        1,
        None,
        zero_copy_send=have_sendfile,
    )
    stream.app_put = AsyncMock()
    return stream


@pytest.mark.parametrize("http_version", ["1.0", "1.1"])
@pytest.mark.anyio
async def test_handle_request_http_1(stream: HTTPStream, http_version: str) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version=http_version,
            headers=[],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    stream.task_group.spawn_app.assert_called()  # type: ignore[attr-defined]
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    # Zero copy send is offered on plaintext HTTP/1.1, but only where os.sendfile exists.
    expected_extensions: dict = {"http.response.pathsend": {}}
    if have_sendfile:
        expected_extensions["http.response.zerocopysend"] = {}
    assert scope == {
        "type": "http",
        "http_version": http_version,
        "asgi": {"spec_version": "2.1", "version": "3.0"},
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"a=b",
        "root_path": stream.config.root_path,
        "headers": [],
        "client": None,
        "server": None,
        "extensions": expected_extensions,
        "state": ConnectionState({}),
    }


@pytest.mark.anyio
async def test_handle_request_http_2(stream: HTTPStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    stream.task_group.spawn_app.assert_called()  # type: ignore[attr-defined]
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    assert scope == {
        "type": "http",
        "http_version": "2",
        "asgi": {"spec_version": "2.1", "version": "3.0"},
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"a=b",
        "root_path": stream.config.root_path,
        "headers": [],
        "client": None,
        "server": None,
        "extensions": {
            "http.response.trailers": {},
            "http.response.early_hint": {},
            "http.response.push": {},
            "http.response.pathsend": {},
        },
        "state": ConnectionState({}),
    }


@pytest.mark.anyio
async def test_handle_request_http_tls() -> None:
    stream = HTTPStream(
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
    stream.app_put = AsyncMock()
    capture_logs(stream.config)
    await stream.handle(
        Request(
            stream_id=1,
            http_version="1.1",
            headers=[],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    assert "tls" in scope["extensions"]
    assert scope["extensions"]["tls"]["client_cert_chain"] == ()
    assert scope["scheme"] == "https"


@pytest.mark.anyio
async def test_handle_body(stream: HTTPStream) -> None:
    await stream.handle(Body(stream_id=1, data=b"data"))
    stream.app_put.assert_called()  # type: ignore[attr-defined]
    assert stream.app_put.call_args_list == [  # type: ignore[attr-defined]
        call({"type": "http.request", "body": b"data", "more_body": True})
    ]


@pytest.mark.anyio
async def test_handle_end_body(stream: HTTPStream) -> None:
    stream.app_put = AsyncMock()
    await stream.handle(EndBody(stream_id=1))
    stream.app_put.assert_called()
    assert stream.app_put.call_args_list == [
        call({"type": "http.request", "body": b"", "more_body": False})
    ]


@pytest.mark.anyio
async def test_handle_closed(stream: HTTPStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.handle(StreamClosed(stream_id=1))
    stream.app_put.assert_called()  # type: ignore[attr-defined]
    assert stream.app_put.call_args_list == [call({"type": "http.disconnect"})]  # type: ignore[attr-defined]


def _get_request() -> Request:
    return Request(
        stream_id=1,
        http_version="1.1",
        headers=[(b"host", b"anycorn")],
        raw_path=b"/",
        method="GET",
        state=ConnectionState({}),
    )


@pytest.mark.anyio
async def test_pathsend_extension_is_advertised(stream: HTTPStream) -> None:
    """Path send is protocol-agnostic, so it is offered on HTTP/1.1 too."""
    await stream.handle(_get_request())
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    assert scope["extensions"]["http.response.pathsend"] == {}


@pytest.mark.anyio
async def test_pathsend_streams_the_named_file(stream: HTTPStream, tmp_path: Path) -> None:
    """A pathsend message streams the file at that path as the response body, then closes."""
    sent: list[Event] = []

    async def send(event: Event) -> None:
        sent.append(event)

    stream.send = send  # a real collector rather than the fixture's mock
    await stream.handle(_get_request())

    payload = b"the quick brown fox\n" * 5000  # larger than PATHSEND_CHUNK_SIZE
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)

    await stream.app_send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", str(len(payload)).encode())],
        }
    )
    pathsend: HTTPResponsePathSendEvent = {
        "type": "http.response.pathsend",
        "path": str(file_path),
    }
    await stream.app_send(pathsend)

    # On HTTP/1.1 the file is handed to the connection as a ZeroCopySend (os.sendfile),
    # so its bytes are not read here; the whole file is covered and the response closed.
    # The bytes actually reaching a client are covered by the socketpair test below.
    zerocopy = next(event for event in sent if isinstance(event, ZeroCopySend))
    assert (zerocopy.offset, zerocopy.count) == (0, len(payload))
    assert any(isinstance(event, EndBody) for event in sent)
    assert any(isinstance(event, StreamClosed) for event in sent)


@pytest.mark.anyio
async def test_send_response(stream: HTTPStream, logs: LogCapture) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "headers": []},
        )
    )
    assert stream.state == ASGIHTTPState.RESPONSE
    await stream.app_send(
        cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": b"Body"})
    )
    assert stream.state == ASGIHTTPState.CLOSED
    stream.send.assert_called()  # type: ignore[unresolved-attribute]
    assert stream.send.call_args_list == [  # type: ignore[unresolved-attribute]
        call(Response(stream_id=1, headers=[], status_code=200)),
        call(Body(stream_id=1, data=b"Body")),
        call(EndBody(stream_id=1)),
        call(StreamClosed(stream_id=1)),
    ]
    # The real access log ran, rather than a mock recording that it was called
    assert len(logs.access) == 1
    assert '"GET / 2" 200' in logs.access[0]


@pytest.mark.anyio
async def test_send_closed_does_not_double_log_on_concurrent_stream_close(
    stream: HTTPStream, logs: LogCapture
) -> None:
    """A StreamClosed racing the response's completion must not log the request twice.

    When the client closes just as the response finalises, the reader task can handle
    StreamClosed while EndBody is still in flight. Unless the stream is already CLOSED
    by then, that path logs the request (response=None) and _send_closed goes on to log
    it again with the full response. Marking CLOSED before the EndBody send closes the
    window; here the race is forced deterministically by handling StreamClosed from
    inside the EndBody send itself.

    https://github.com/pgjones/hypercorn/issues/357
    """
    await stream.handle(
        Request(
            stream_id=1,
            http_version="1.1",
            headers=[],
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "headers": []},
        )
    )

    async def _close_during_end_body(event: Event) -> None:
        # The reader task running mid-send, exactly as the race schedules it.
        if isinstance(event, EndBody):
            await stream.handle(StreamClosed(stream_id=1))

    stream.send = AsyncMock(side_effect=_close_during_end_body)

    await stream.app_send(
        cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": b"Body"})
    )

    assert len(logs.access) == 1


@pytest.mark.anyio
async def test_invalid_server_name(stream: HTTPStream) -> None:
    stream.config.server_names = ["anycorn"]
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"host", b"example.com")],
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
async def test_send_push(stream: HTTPStream, http_scope: HTTPScope) -> None:
    stream.scope = http_scope
    stream.stream_id = 1
    await stream.app_send({"type": "http.response.push", "path": "/push", "headers": []})
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(
            Request(
                stream_id=1,
                headers=[(b":scheme", b"https")],
                http_version="2",
                method="GET",
                raw_path=b"/push",
                state=ConnectionState({}),
            )
        )
    ]


@pytest.mark.anyio
async def test_send_early_hint(stream: HTTPStream, http_scope: HTTPScope) -> None:
    stream.scope = http_scope
    stream.stream_id = 1
    await stream.app_send(
        {"type": "http.response.early_hint", "links": [b'</style.css>; rel="preload"; as="style"']}
    )
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(
            InformationalResponse(
                stream_id=1,
                headers=[(b"link", b'</style.css>; rel="preload"; as="style"')],
                status_code=103,
            )
        )
    ]


@pytest.mark.anyio
async def test_send_trailers(stream: HTTPStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[(b"te", b"trailers")],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "trailers": True},
        )
    )
    await stream.app_send(
        cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": b"Body"})
    )
    await stream.app_send({"type": "http.response.trailers", "headers": [(b"X", b"V")]})
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(Response(stream_id=1, headers=[], status_code=200)),
        call(Body(stream_id=1, data=b"Body")),
        call(Trailers(stream_id=1, headers=[(b"X", b"V")])),
        call(EndBody(stream_id=1)),
        call(StreamClosed(stream_id=1)),
    ]


@pytest.mark.anyio
async def test_send_trailers_ignored(stream: HTTPStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],  # no TE: trailers header
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "trailers": True},
        )
    )
    await stream.app_send(
        cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": b"Body"})
    )
    await stream.app_send({"type": "http.response.trailers", "headers": [(b"X", b"V")]})
    assert stream.send.call_args_list == [  # type: ignore[attr-defined]
        call(Response(stream_id=1, headers=[], status_code=200)),
        call(Body(stream_id=1, data=b"Body")),
        call(EndBody(stream_id=1)),
        call(StreamClosed(stream_id=1)),
    ]


@pytest.mark.anyio
async def test_send_app_error(stream: HTTPStream, logs: LogCapture) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],
            raw_path=b"/?a=b",
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
    assert '"GET / 2" 500' in logs.access[0]


@pytest.mark.parametrize(
    ("state", "message_type"),
    [
        (ASGIHTTPState.REQUEST, "not_a_real_type"),
        (ASGIHTTPState.RESPONSE, "http.response.start"),
        (ASGIHTTPState.TRAILERS, "http.response.start"),
        (ASGIHTTPState.CLOSED, "http.response.start"),
        (ASGIHTTPState.CLOSED, "http.response.body"),
        (ASGIHTTPState.CLOSED, "http.response.trailers"),
    ],
)
@pytest.mark.anyio
async def test_send_invalid_message_given_state(
    stream: HTTPStream, state: ASGIHTTPState, http_scope: HTTPScope, message_type: str
) -> None:
    stream.state = state
    stream.scope = http_scope
    with pytest.raises(UnexpectedMessageError):
        await stream.app_send({"type": message_type})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "headers", "body"),
    [
        ("201 NO CONTENT", [], b""),  # Status should be int
        (200, [("X-Foo", "foo")], b""),  # Headers should be bytes
        (200, [], "Body"),  # Body should be bytes
    ],
)
@pytest.mark.anyio
async def test_send_invalid_message(
    stream: HTTPStream,
    http_scope: HTTPScope,
    status: Any,  # noqa: ANN401
    headers: Any,  # noqa: ANN401
    body: Any,  # noqa: ANN401
) -> None:
    stream.scope = http_scope
    stream.state = ASGIHTTPState.REQUEST
    with pytest.raises((TypeError, ValueError)):  # noqa: PT012
        await stream.app_send(
            cast(
                "HTTPResponseStartEvent",
                {"type": "http.response.start", "headers": headers, "status": status},
            )
        )
        await stream.app_send(
            cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": body})
        )


@pytest.mark.anyio
async def test_stream_idle(stream: HTTPStream) -> None:
    assert stream.idle is False


@pytest.mark.anyio
async def test_closure(stream: HTTPStream) -> None:
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],
            raw_path=b"/?a=b",
            method="GET",
            state=ConnectionState({}),
        )
    )
    assert not stream.closed
    await stream.handle(StreamClosed(stream_id=1))
    assert stream.closed
    await stream.handle(StreamClosed(stream_id=1))
    assert stream.closed
    # It is important that the disconnect message has only been sent
    # once.
    assert stream.app_put.call_args_list == [call({"type": "http.disconnect"})]  # type: ignore[unresolved-attribute]


@pytest.mark.anyio
async def test_abnormal_close_logging() -> None:
    config = Config()
    config.accesslog = "-"
    config.statsd_host = "localhost:9125"
    # This exercises an issue where `HTTPStream` at one point called the statsd logger
    # with `response=None` when the statsd logger failed to handle it.
    config.set_statsd_logger_class(StatsdLogger)
    stream = HTTPStream(
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

    async with config.log:
        await stream.handle(
            Request(
                stream_id=1,
                http_version="2",
                headers=[],
                raw_path=b"/?a=b",
                method="GET",
                state=ConnectionState({}),
            )
        )
        await stream.handle(StreamClosed(stream_id=1))


@pytest.mark.anyio
async def test_trailers_without_te_do_not_crash(stream: HTTPStream) -> None:
    """Trailers as the first message, from a client that never asked for them.

    Nothing is sent - the client did not offer `te: trailers` - but closing the
    stream here used to read self.response, which no response had yet assigned,
    so the app got an AttributeError rather than a reply.
    """
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],  # no te: trailers
            raw_path=b"/",
            method="GET",
            state=ConnectionState({}),
        )
    )

    await stream.app_send({"type": "http.response.trailers", "headers": [(b"x", b"y")]})

    # Still awaiting a response rather than closed, so the app can go on to send one
    assert stream.state == ASGIHTTPState.REQUEST


def _request(http_version: str = "1.1", *, method: str = "GET", headers: Any = None) -> Request:  # noqa: ANN401
    return Request(
        stream_id=1,
        http_version=http_version,
        headers=headers if headers is not None else [(b"host", b"anycorn")],
        raw_path=b"/",
        method=method,
        state=ConnectionState({}),
    )


def _collect(stream: HTTPStream) -> list[Event]:
    """Replace the stream's send with a real collector and return the list it fills."""
    sent: list[Event] = []

    async def send(event: Event) -> None:
        sent.append(event)

    stream.send = send
    return sent


@pytest.mark.anyio
async def test_handle_ignores_an_unexpected_event(stream: HTTPStream) -> None:
    """Handle only reacts to Request/Body/EndBody/StreamClosed; anything else is a no-op."""
    sent = _collect(stream)
    await stream.handle(Trailers(stream_id=1, headers=[]))
    assert sent == []


@pytest.mark.anyio
async def test_push_with_a_non_str_path_is_rejected(
    stream: HTTPStream, http_scope: HTTPScope
) -> None:
    """A push message must name its path as a str."""
    stream.scope = http_scope
    with pytest.raises(TypeError, match="should be a str"):
        await stream.app_send(
            cast("Any", {"type": "http.response.push", "path": 123, "headers": []})
        )


@pytest.mark.anyio
async def test_zerocopysend_defaults_offset_and_count(stream: HTTPStream, tmp_path: Path) -> None:
    """Without offset/count, the whole file from its current position is sent."""
    payload = b"zero-copy" * 100
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    sent = _collect(stream)
    await stream.handle(_request("1.1"))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            },
        )
    )
    with file_path.open("rb") as file:
        await stream.app_send(cast("Any", {"type": "http.response.zerocopysend", "file": file}))
    zerocopy = next(event for event in sent if isinstance(event, ZeroCopySend))
    assert (zerocopy.offset, zerocopy.count) == (0, len(payload))


@pytest.mark.anyio
async def test_zerocopysend_with_more_body_does_not_close(
    stream: HTTPStream, tmp_path: Path
) -> None:
    """more_body keeps the response open rather than finishing it."""
    payload = b"chunk"
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    sent = _collect(stream)
    await stream.handle(_request("1.1"))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent", {"type": "http.response.start", "status": 200, "headers": []}
        )
    )
    with file_path.open("rb") as file:
        await stream.app_send(
            cast(
                "Any",
                {"type": "http.response.zerocopysend", "file": file, "more_body": True},
            )
        )
    assert not any(isinstance(event, EndBody) for event in sent)
    assert stream.state == ASGIHTTPState.RESPONSE


@pytest.mark.anyio
async def test_zerocopysend_with_trailers_transitions_to_trailers_state(
    stream: HTTPStream, tmp_path: Path
) -> None:
    """A body sent with trailers pending moves to the TRAILERS state instead of closing."""
    payload = b"body"
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    _collect(stream)
    await stream.handle(_request("2", headers=[(b"te", b"trailers")]))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "headers": [], "trailers": True},
        )
    )
    with file_path.open("rb") as file:
        await stream.app_send(cast("Any", {"type": "http.response.zerocopysend", "file": file}))
    assert stream.state == ASGIHTTPState.TRAILERS


@pytest.mark.anyio
async def test_pathsend_on_http2_reads_and_frames_the_body(
    stream: HTTPStream, tmp_path: Path
) -> None:
    """On HTTP/2 the file cannot be handed to os.sendfile, so it is read and framed."""
    payload = b"h2 pathsend body\n" * 500
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    sent = _collect(stream)
    await stream.handle(_request("2"))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            },
        )
    )
    await stream.app_send(cast("Any", {"type": "http.response.pathsend", "path": str(file_path)}))
    body = b"".join(event.data for event in sent if isinstance(event, Body))
    assert body == payload  # read and framed as Body chunks, no ZeroCopySend
    assert not any(isinstance(event, ZeroCopySend) for event in sent)


@pytest.mark.anyio
async def test_pathsend_on_a_head_request_suppresses_the_body(
    stream: HTTPStream, tmp_path: Path
) -> None:
    """A HEAD response sends no body, but the file send still closes the stream."""
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(b"x" * 100)
    sent = _collect(stream)
    await stream.handle(_request("1.1", method="HEAD"))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent", {"type": "http.response.start", "status": 200, "headers": []}
        )
    )
    await stream.app_send(cast("Any", {"type": "http.response.pathsend", "path": str(file_path)}))
    assert not any(isinstance(event, (Body, ZeroCopySend)) for event in sent)
    assert any(isinstance(event, EndBody) for event in sent)


@pytest.mark.anyio
async def test_trailers_as_first_message_starts_the_response(stream: HTTPStream) -> None:
    """Trailers sent first, with te: trailers offered, open a 200 response and close it."""
    sent = _collect(stream)
    # A non-te header before te: trailers exercises the header scan skipping past it.
    await stream.handle(_request("2", headers=[(b"host", b"anycorn"), (b"te", b"trailers")]))
    await stream.app_send(
        cast("Any", {"type": "http.response.trailers", "headers": [(b"x", b"y")]})
    )
    responses = [event for event in sent if isinstance(event, Response)]
    assert [event.status_code for event in responses] == [200]
    assert any(isinstance(event, EndBody) for event in sent)
    assert stream.state == ASGIHTTPState.CLOSED


@pytest.mark.anyio
async def test_send_trailers_skips_headers_other_than_te(stream: HTTPStream) -> None:
    """The te: trailers header is found even when other headers come before it."""
    sent = _collect(stream)
    await stream.handle(_request("2", headers=[(b"host", b"anycorn"), (b"te", b"trailers")]))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "headers": [], "trailers": True},
        )
    )
    await stream.app_send(
        cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": b"Body"})
    )
    await stream.app_send(
        cast("Any", {"type": "http.response.trailers", "headers": [(b"X", b"V")]})
    )
    assert any(isinstance(event, Trailers) for event in sent)


@pytest.mark.anyio
async def test_more_trailers_keeps_the_stream_open(stream: HTTPStream) -> None:
    """A trailers message with more_trailers set does not finish the response."""
    sent = _collect(stream)
    await stream.handle(_request("2", headers=[(b"te", b"trailers")]))
    await stream.app_send(
        cast(
            "HTTPResponseStartEvent",
            {"type": "http.response.start", "status": 200, "headers": [], "trailers": True},
        )
    )
    await stream.app_send(
        cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": b"Body"})
    )
    await stream.app_send(
        cast(
            "Any",
            {"type": "http.response.trailers", "headers": [(b"X", b"V")], "more_trailers": True},
        )
    )
    assert not any(isinstance(event, EndBody) for event in sent)
    assert stream.state == ASGIHTTPState.TRAILERS


@pytest.mark.parametrize(
    ("raw_path", "expected"),
    [
        (b"/caf%C3%A9", "/café"),  # percent-encoded UTF-8, decoded per the ASGI spec
        (b"/a%20b", "/a b"),
        (b"/x%2Fy", "/x/y"),
    ],
)
@pytest.mark.anyio
async def test_handle_request_percent_encoded_path(
    stream: HTTPStream, raw_path: bytes, expected: str
) -> None:
    """A valid escape sequence is decoded into the character it stands for."""
    await stream.handle(
        Request(
            stream_id=1,
            http_version="2",
            headers=[],
            raw_path=raw_path,
            method="GET",
            state=ConnectionState({}),
        )
    )
    scope = stream.task_group.spawn_app.call_args[0][2]  # type: ignore[attr-defined]
    assert scope["path"] == expected
    # The undecoded bytes stay available for apps that need them
    assert scope["raw_path"] == raw_path
