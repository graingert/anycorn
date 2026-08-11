"""TCP server implementation for handling incoming connections."""

from __future__ import annotations

import contextlib
from math import inf
from socket import AF_INET, AF_INET6
from ssl import SSLZeroReturnError
from typing import TYPE_CHECKING

import anyio
import anyio.abc
import anyio.streams.tls

from .events import Closed, Event, RawData, SendFile, Updated
from .ktls import KTLSAttribute
from .protocol import ProtocolWrapper
from .sendfile import have_sendfile, read_file_chunks, sendfile
from .task_group import TaskGroup
from .typing import AppWrapper, ConnectionState, LifespanState
from .utils import build_tls_extension, parse_socket_addr
from .worker_context import AnyioSingleTask, WorkerContext

if TYPE_CHECKING:
    import socket as socket_module
    from collections.abc import Callable

    from .config import Config

MAX_RECV = 2**16

# os.sendfile is only used over TCP: it is unsupported over a UNIX domain socket on macOS,
# and kTLS (the TLS send path) is TCP-only, so a UNIX-socket connection reads and sends
# through the stream instead.
_SENDFILE_FAMILIES = frozenset({AF_INET, AF_INET6})


class TCPServer:
    """Handles a single TCP connection, managing protocol negotiation and I/O."""

    def __init__(
        self,
        app: AppWrapper,
        config: Config,
        context: WorkerContext,
        state: LifespanState,
        stream: anyio.abc.SocketStream,
    ) -> None:
        self.app = app
        self.config = config
        self.context = context
        self.protocol: ProtocolWrapper
        self.send_lock = anyio.Lock()
        self.idle_task = AnyioSingleTask()
        self.state = state
        self.stream = stream

        self._idle_handle: anyio.CancelScope | None = None
        # The raw socket to os.sendfile over, set once the connection is known to be
        # plaintext (a TLS stream encrypts in userspace, so it cannot be zero copy).
        self._sendfile_socket: socket_module.socket | None = None

    async def run(self) -> None:
        """Run the server for this connection."""
        try:
            alpn_protocol = self.stream.extra(anyio.streams.tls.TLSAttribute.alpn_protocol)  # noqa: S610
            tls_extension = build_tls_extension(self.config, self.stream)
        except anyio.TypedAttributeLookupError:  # Not SSL
            alpn_protocol = "http/1.1"
            tls_extension = None

        try:
            socket = self.stream.extra(anyio.abc.SocketAttribute.raw_socket)  # noqa: S610
            client = parse_socket_addr(socket.family, socket.getpeername())
            server = parse_socket_addr(socket.family, socket.getsockname())
            # Real zero-copy send needs a socket the kernel can put the file bytes on
            # directly: a plaintext socket, or a TLS one the kernel encrypts itself (kTLS).
            # Userspace TLS encrypts in Python, so the body is read and sent through the
            # stream instead. Only TCP is used: os.sendfile is not portable over a UNIX
            # domain socket (it is unsupported on macOS), and kTLS is TCP-only anyway.
            if have_sendfile and socket.family in _SENDFILE_FAMILIES:
                if tls_extension is None:  # pragma: win32 no cover - no os.sendfile on Windows
                    self._sendfile_socket = socket
                else:
                    self._sendfile_socket = self._ktls_sendfile_socket()

            async with TaskGroup() as task_group:
                self._task_group = task_group
                self.protocol = ProtocolWrapper(
                    self.app,
                    self.config,
                    self.context,
                    task_group,
                    ConnectionState(self.state.copy()),
                    client,
                    server,
                    self.protocol_send,
                    tls_extension,
                    alpn_protocol,
                    zero_copy_send=self._sendfile_socket is not None,
                )
                await self.protocol.initiate()
                await self.idle_task.restart(self._task_group, self._idle_timeout)
                await self._read_data()
        except OSError:
            pass
        finally:
            await self._close()

    def _ktls_sendfile_socket(  # pragma: win32 no cover - reached only on the sendfile path
        self,
    ) -> socket_module.socket | None:
        """Return the socket to os.sendfile over when the kernel owns TLS send, else None.

        Only a KTLSStream exposes this attribute, and only when kTLS actually activated;
        an ordinary userspace TLS stream has no such attribute, so this stays None and the
        body is read and encrypted through the stream as before.
        """
        try:
            return self.stream.extra(KTLSAttribute.sendfile_socket)  # noqa: S610
        except anyio.TypedAttributeLookupError:
            return None

    async def protocol_send(self, event: Event) -> None:
        """Forward a protocol event to the underlying stream."""
        if isinstance(event, RawData):
            async with self.send_lock:
                try:
                    with anyio.CancelScope(shield=True):
                        await self.stream.send(event.data)
                except (anyio.ClosedResourceError, anyio.BrokenResourceError, TimeoutError):
                    await self.protocol.handle(Closed())
        elif isinstance(event, SendFile):
            async with self.send_lock:
                try:
                    with anyio.CancelScope(shield=True):
                        await self._transmit_file(event)
                except (
                    anyio.ClosedResourceError,
                    anyio.BrokenResourceError,
                    TimeoutError,
                    OSError,
                ):
                    await self.protocol.handle(Closed())
        elif isinstance(event, Closed):
            await self._close()
            await self.protocol.handle(Closed())
        elif isinstance(event, Updated):
            if event.idle:
                await self.idle_task.restart(self._task_group, self._idle_timeout)
            else:
                await self.idle_task.stop()

    async def _transmit_file(self, event: SendFile) -> None:
        """Send a file body, with a zero-copy os.sendfile where the socket allows it."""
        if self._sendfile_socket is not None:
            # No os.sendfile on Windows, so _sendfile_socket is never set there.
            await sendfile(  # pragma: win32 no cover
                self._sendfile_socket, event.file, event.offset, event.count
            )
        else:
            # A TLS connection (or a platform without sendfile): read the window and send
            # it through the stream, which encrypts it as any other body bytes.
            async for chunk in read_file_chunks(event.file, event.offset, event.count):
                await self.stream.send(chunk)

    async def _read_data(self) -> None:
        while True:
            try:
                with anyio.fail_after(self.config.read_timeout or inf):
                    data = await self.stream.receive(MAX_RECV)
            except (  # noqa: PERF203
                anyio.ClosedResourceError,
                anyio.BrokenResourceError,
                anyio.EndOfStream,
                TimeoutError,
                SSLZeroReturnError,
            ):
                break
            else:
                await self.protocol.handle(RawData(data))
                if data == b"":
                    break
        await self.protocol.handle(Closed())

    async def _close(self) -> None:
        with contextlib.suppress(
            OSError,
            anyio.BrokenResourceError,
            AttributeError,
            NotImplementedError,
            TypeError,
            anyio.BusyResourceError,
            anyio.ClosedResourceError,
        ):
            # They're already gone, nothing to do, or it is a SSL stream
            await self.stream.send_eof()
        with contextlib.suppress(
            OSError, anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.BusyResourceError
        ):
            # OSError (which SSLError subclasses) for the same reason send_eof above
            # suppresses it: an abrupt disconnect can surface a network-unreachable
            # errno - EHOSTUNREACH/ENETUNREACH/ETIMEDOUT - that asyncio never maps to
            # ConnectionError, so it would otherwise escape _close() and, since this
            # runs from run()'s finally and from protocol_send, crash the connection
            # task or propagate back into the app (https://github.com/pgjones/hypercorn/issues/361).
            await self.stream.aclose()

    async def _idle_timeout(self) -> None:
        with self.context.move_on_after(self.config.keep_alive_timeout):
            await self.context.terminated.wait()

        with anyio.CancelScope(shield=True):
            await self._initiate_server_close()

    async def _initiate_server_close(self) -> None:
        await self.protocol.handle(Closed())
        # OSError (SSLError included) as in _close: aclose on an unreachable peer can
        # raise a plain OSError that would otherwise escape the idle-timeout task.
        with contextlib.suppress(OSError, anyio.BrokenResourceError, anyio.BusyResourceError):
            await self.stream.aclose()


def tcp_server_handler(
    app: AppWrapper,
    config: Config,
    context: WorkerContext,
    state: LifespanState,
) -> Callable:
    """Return a handler callable suitable for use with anyio's listener.serve()."""

    async def handler(stream: anyio.abc.SocketStream) -> None:
        tcp_server = TCPServer(app, config, context, state, stream)
        await tcp_server.run()

    return handler
