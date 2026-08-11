"""Tests for TCPServer connection teardown."""

from __future__ import annotations

import errno
import socket
from math import inf
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import anyio
import h11
import pytest
from anyio.abc import SocketAttribute

from anycorn.app_wrappers import ASGIWrapper
from anycorn.config import Config
from anycorn.events import Closed, Event, RawData, SendFile
from anycorn.tcp_server import TCPServer
from anycorn.worker_context import WorkerContext

from .helpers import (
    SANITY_BODY,
    MemoryClientStream,
    MemorySocketStream,
    MockSocket,
    memory_socket_stream_pair,
    sanity_framework,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


def _server(stream: MemorySocketStream) -> TCPServer:
    return TCPServer(ASGIWrapper(sanity_framework), Config(), WorkerContext(None), {}, stream)


def _stream_pair(
    server_cls: type[MemorySocketStream] = MemorySocketStream,
    *,
    attributes: Mapping[Any, Callable[[], Any]] | None = None,
) -> tuple[MemoryClientStream, MemorySocketStream]:
    """Return a client/server memory pair, varying the server stream class and attributes."""
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream[bytes](inf)
    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream[bytes](inf)
    attrs = attributes if attributes is not None else {SocketAttribute.raw_socket: MockSocket}
    return (
        MemoryClientStream(server_to_client_receive, client_to_server_send),
        server_cls(client_to_server_receive, server_to_client_send, attrs),
    )


class _UnreachableOnCloseStream(MemorySocketStream):
    """A stream whose close raises a network-unreachable OSError.

    asyncio maps only a handful of errno (ECONNRESET, EPIPE, ...) to ConnectionError;
    EHOSTUNREACH and its kin (ENETUNREACH, ETIMEDOUT) stay plain OSError, as an abrupt
    client disconnect - a deleted pod, a pulled cable - can produce.
    """

    def __init__(
        self,
        receive_stream: MemoryObjectReceiveStream[bytes],
        send_stream: MemoryObjectSendStream[bytes],
        attributes: Mapping[Any, Callable[[], Any]],
    ) -> None:
        super().__init__(receive_stream, send_stream, attributes)
        self.aclose_called = False

    async def aclose(self) -> None:
        self.aclose_called = True
        await super().aclose()
        raise OSError(errno.EHOSTUNREACH, "No route to host")


@pytest.mark.anyio
async def test_a_connection_whose_close_fails_does_not_crash_the_server() -> None:
    """The teardown OSError must not escape run(), which is where it would land.

    _close() runs from run()'s finally - outside its own `except OSError` - and from
    protocol_send, so an escaping OSError crashes the connection task or propagates
    back into the ASGI app (https://github.com/pgjones/hypercorn/issues/361). Driving
    a whole request through means run() really is what has to survive it, rather than
    _close() being called directly and the caller assumed.
    """
    client_stream, plain_server_stream = memory_socket_stream_pair()
    server_stream = _UnreachableOnCloseStream(
        plain_server_stream._receive_stream,
        plain_server_stream._send_stream,
        plain_server_stream.extra_attributes,
    )
    server = TCPServer(
        ASGIWrapper(sanity_framework), Config(), WorkerContext(None), {}, server_stream
    )

    client = h11.Connection(h11.CLIENT)
    statuses = []
    # Only the client end is closed here: closing the server end is the server's job,
    # and this one raises by design, so doing it again would be the test failing itself
    async with client_stream:
        # An escaping OSError propagates out of the task group, which is the assertion
        with anyio.fail_after(5):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(server.run)

                await client_stream.send_all(
                    client.send(
                        h11.Request(
                            method="POST",
                            target="/",
                            headers=[
                                (b"host", b"anycorn"),
                                (b"connection", b"close"),
                                (b"content-length", b"%d" % len(SANITY_BODY)),
                            ],
                        )
                    )
                )
                await client_stream.send_all(client.send(h11.Data(data=SANITY_BODY)))
                await client_stream.send_all(client.send(h11.EndOfMessage()))

                while True:
                    event = client.next_event()
                    if event is h11.NEED_DATA:
                        client.receive_data(await client_stream.receive_some(4096))
                    elif isinstance(event, h11.Response):
                        statuses.append(event.status_code)
                    elif isinstance(event, h11.ConnectionClosed):
                        break

                # _read_data stays parked on the memory stream once it is closed, as
                # serve_in_memory also has to allow for, so stop the server rather
                # than wait on a run() that will not return
                task_group.cancel_scope.cancel()

    # The request was served, and the failing teardown did not take run() down with it
    assert statuses == [200]
    assert server_stream.aclose_called


class _SendBrokenStream(MemorySocketStream):
    """A stream whose send always reports the peer has gone."""

    async def send(self, item: bytes) -> None:  # noqa: ARG002
        raise anyio.BrokenResourceError


@pytest.mark.anyio
async def test_protocol_send_rawdata_closes_when_the_stream_is_broken() -> None:
    """A failed raw send tells the protocol the connection is gone rather than raising."""
    client, server_stream = _stream_pair(_SendBrokenStream)
    server = _server(server_stream)
    server.protocol = AsyncMock()
    await server.protocol_send(RawData(data=b"hi"))
    server.protocol.handle.assert_awaited_once_with(Closed())
    await client.aclose()
    await server_stream.aclose()


@pytest.mark.anyio
async def test_protocol_send_sendfile_closes_when_the_transmit_fails() -> None:
    """A failed file transmit (here an invalid fd) is turned into a Closed, not an error."""
    client, server_stream = _stream_pair()
    server = _server(server_stream)
    server.protocol = AsyncMock()
    # No sendfile socket set, so the read path runs; an invalid descriptor makes it raise
    # OSError, which protocol_send must catch.
    await server.protocol_send(SendFile(file=-1, offset=0, count=10))
    server.protocol.handle.assert_awaited_once_with(Closed())
    await client.aclose()
    await server_stream.aclose()


@pytest.mark.anyio
async def test_protocol_send_sendfile_reads_and_sends_through_the_stream(tmp_path: Path) -> None:
    """Without a sendfile socket (TLS, or Windows), the file window is read and sent."""
    payload = b"read-path body\n" * 100
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    client, server_stream = _stream_pair()
    server = _server(server_stream)
    server.protocol = AsyncMock()
    sent = bytearray()

    async def collect() -> None:
        while True:
            chunk = await client.receive_some()
            if not chunk:
                return
            sent.extend(chunk)

    with file_path.open("rb") as file:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(collect)
            await anyio.wait_all_tasks_blocked()
            await server.protocol_send(SendFile(file=file.fileno(), offset=0, count=len(payload)))
            await server_stream.aclose()  # EOF so collect finishes
    server.protocol.handle.assert_not_awaited()  # a successful send, no Closed
    assert bytes(sent) == payload
    await client.aclose()


@pytest.mark.anyio
async def test_protocol_send_ignores_an_unknown_event() -> None:
    """protocol_send only reacts to RawData/SendFile/Closed/Updated; anything else is a no-op."""
    client, server_stream = _stream_pair()
    server = _server(server_stream)
    server.protocol = AsyncMock()
    await server.protocol_send(Event())
    server.protocol.handle.assert_not_awaited()
    await client.aclose()
    await server_stream.aclose()


class _EmptyReadStream(MemorySocketStream):
    """A stream that reads back an empty byte string, as a half-closed socket does."""

    async def receive(self, max_bytes: int = 65536) -> bytes:  # noqa: ARG002
        return b""  # a real socket signals a peer half-close with an empty read


@pytest.mark.anyio
async def test_an_empty_read_ends_the_read_loop() -> None:
    """An empty read (peer half-close) stops the receive loop and closes the connection."""
    client, server_stream = _stream_pair(_EmptyReadStream)
    server = _server(server_stream)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(server.run)
            # The empty read breaks the loop and closes the connection; the server then
            # parks on its idle task, so stop it rather than wait on keep-alive.
            await anyio.wait_all_tasks_blocked()
            await client.aclose()
            task_group.cancel_scope.cancel()
    await server_stream.aclose()


class _PeerNameRaises:
    """A raw socket whose getpeername raises, as an already-dropped connection can."""

    family = socket.AF_INET

    def getpeername(self) -> tuple[str, int]:  # read first in run(), before getsockname
        raise OSError(errno.ENOTCONN, "socket is not connected")


@pytest.mark.anyio
async def test_run_swallows_an_oserror_while_setting_up() -> None:
    """An OSError reading the peer address must be caught by run, not escape it."""
    client, server_stream = _stream_pair(
        attributes={SocketAttribute.raw_socket: _PeerNameRaises}
    )
    server = _server(server_stream)
    with anyio.fail_after(5):
        await server.run()  # the OSError is swallowed and teardown still runs
    await client.aclose()


_AF_UNIX = getattr(socket, "AF_UNIX", None)


class _UnixSocket:
    """A raw socket over a UNIX domain, where os.sendfile is not used."""

    family = _AF_UNIX

    def getsockname(self) -> str:
        return "/tmp/anycorn-server.sock"  # noqa: S108

    def getpeername(self) -> str:
        return "/tmp/anycorn-client.sock"  # noqa: S108


@pytest.mark.skipif(_AF_UNIX is None, reason="AF_UNIX is unavailable")
@pytest.mark.anyio
async def test_a_unix_socket_connection_does_not_arm_sendfile() -> None:
    """os.sendfile is TCP-only here, so a UNIX-family connection keeps the read path."""
    client, server_stream = _stream_pair(attributes={SocketAttribute.raw_socket: _UnixSocket})
    server = _server(server_stream)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(server.run)
        await anyio.wait_all_tasks_blocked()
        assert server._sendfile_socket is None  # not armed for a non-INET family
        await client.aclose()  # EOF so run finishes
        task_group.cancel_scope.cancel()
    await server_stream.aclose()
