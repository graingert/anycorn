"""A socket-backed TLS stream that can enable kernel TLS (kTLS) for zero-copy send.

anyio - like asyncio and trio - does TLS with an in-memory BIO: OpenSSL encrypts in
Python and the ciphertext is written to a plaintext transport. That makes ``os.sendfile``
impossible over TLS, because the file bytes would go out unencrypted. kTLS moves record
encryption into the kernel, so ``sendfile`` on the socket comes out encrypted - but it
requires the TLS session to be bound to a real socket fd (``wrap_socket``, not
``wrap_bio``), which no async framework does out of the box.

This stream fills that gap: it drives a non-blocking ``SSLSocket`` directly, awaiting
readability/writability on ``WouldBlock``, and asks OpenSSL to enable kTLS. When the
kernel actually takes over the send path, :attr:`sendfile_socket` exposes the socket so
the body can be sent with ``os.sendfile``; otherwise the stream still works as ordinary
userspace TLS and callers fall back to reading and sending.

kTLS is Linux-only and depends on the kernel ``tls`` ULP, an OpenSSL built with kTLS, a
compatible cipher, and a Python whose ``ssl``/``socket`` modules expose the constants. All
of that is capability-detected; where any piece is missing this is plain socket-backed TLS.
"""

from __future__ import annotations

import contextlib
import logging
import ssl
import sys
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import anyio
import anyio.abc
from anyio import BrokenResourceError, EndOfStream, TypedAttributeSet, typed_attribute
from anyio.abc import SocketAttribute
from anyio.streams.tls import TLSAttribute

if TYPE_CHECKING:
    import socket
    from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)

# ``SOL_TLS`` and ``TLS_TX`` are part of the kernel TLS ABI (uapi/linux/tls.h) but CPython's
# ``socket`` module does not export them, so they are pinned here to their stable Linux values.
# They are only meaningful on Linux with the ``tls`` ULP; every use is guarded by a
# ``getsockopt`` that fails cleanly where the ULP is absent.
_SOL_TLS = 282
_TLS_TX = 1
# ``sizeof(struct tls_crypto_info)`` - the base header the kernel copies out for a TLS_TX
# getsockopt. The kernel rejects a shorter buffer with EINVAL *before* it checks whether the
# send path is configured, so the probe must request at least this many bytes.
_TLS_CRYPTO_INFO_SIZE = 4

# ``ssl.OP_ENABLE_KTLS`` exists only when Python is linked against an OpenSSL built with kTLS
# support; it is the one piece that cannot be assumed. kTLS itself is Linux-only.
_OP_ENABLE_KTLS = getattr(ssl, "OP_ENABLE_KTLS", None)
can_enable_ktls = _OP_ENABLE_KTLS is not None and sys.platform == "linux"


def enable_ktls(context: ssl.SSLContext) -> None:  # pragma: linux cover
    """Ask an SSL context to enable kTLS, if this build of OpenSSL/Python can.

    A no-op where kTLS is unavailable, so it is always safe to call - the resulting
    connection is then ordinary userspace TLS.
    """
    # Whether this branch is taken is fixed by the OpenSSL build, so only one arm is ever
    # reachable on a given job (kTLS builds take the body, others fall through).
    if _OP_ENABLE_KTLS is not None:  # pragma: no branch
        context.options |= _OP_ENABLE_KTLS  # pragma: >=3.12 cover


def _ktls_send_active(sock: socket.socket) -> bool:  # pragma: linux cover
    """Return True only when the kernel has taken over the TLS send path for *sock*.

    Reads the ``TLS_TX`` socket option: the kernel only answers it once OpenSSL has
    attached the ``tls`` ULP and installed the send keys, so a successful read is a
    positive confirmation. Any error - the ULP absent (the level is unknown, ENOPROTOOPT),
    the send path not configured (EBUSY), a wrong platform - is treated as "not active", so
    a plaintext ``sendfile`` is never done over a connection still encrypting in userspace.
    """
    # can_enable_ktls is constant for the process, so only one arm runs on a given job.
    if not can_enable_ktls:  # pragma: no branch
        return False
    try:
        sock.getsockopt(_SOL_TLS, _TLS_TX, _TLS_CRYPTO_INFO_SIZE)
    except OSError:
        return False
    # A successful read means the kernel owns the send path, which only a real kTLS
    # socket (Linux 3.12+ with the tls ULP) ever provides.
    return True  # pragma: >=3.12 cover


class KTLSStream(anyio.abc.ByteStream):  # pragma: linux cover
    """A byte stream doing TLS on a real socket, with kTLS zero-copy send when available."""

    def __init__(self, ssl_sock: ssl.SSLSocket) -> None:
        self._ssl_sock = ssl_sock
        self._raw_socket = ssl_sock
        # A plaintext sendfile is only safe once the kernel owns the send crypto, which only
        # a real kTLS socket (Linux 3.12+) reports; elsewhere this stays None.
        self._sendfile_socket: socket.socket | None = (
            ssl_sock if _ktls_send_active(ssl_sock) else None  # pragma: >=3.12 cover
        )

    @property
    def sendfile_socket(self) -> socket.socket | None:
        """The socket to os.sendfile over, or None when send is still userspace TLS."""
        return self._sendfile_socket

    @classmethod
    async def wrap(
        cls,
        sock: socket.socket,
        *,
        ssl_context: ssl.SSLContext,
        server_side: bool = True,
        server_hostname: str | None = None,
    ) -> KTLSStream:
        """Perform the TLS handshake over *sock* (non-blocking) and return the stream."""
        sock.setblocking(False)  # noqa: FBT003
        ssl_sock = ssl_context.wrap_socket(
            sock,
            server_side=server_side,
            server_hostname=server_hostname,
            do_handshake_on_connect=False,
        )
        stream = cls(ssl_sock)
        try:
            await stream._retry(ssl_sock.do_handshake)
        except BaseException:
            # wrap_socket detached the raw socket, so ssl_sock now owns the fd; close it here
            # or a failed handshake leaks it until GC (which then errors on the dead fd).
            with contextlib.suppress(OSError):
                ssl_sock.close()
            raise
        # kTLS is negotiated during the handshake, so only now is the send path known. The
        # kernel-owned send side only exists on a real kTLS socket (Linux 3.12+).
        stream._sendfile_socket = (
            ssl_sock if _ktls_send_active(ssl_sock) else None  # pragma: >=3.12 cover
        )
        return stream

    async def _retry(self, func: Callable[..., Any], *args: Any) -> Any:  # noqa: ANN401
        """Run a blocking SSL call, awaiting readability/writability until it completes."""
        while True:
            try:
                return func(*args)
            except ssl.SSLWantReadError:  # noqa: PERF203
                await anyio.wait_readable(self._ssl_sock)
            except ssl.SSLWantWriteError:  # pragma: no cover
                # Only seen when the socket send buffer fills mid-write, which a test
                # cannot force deterministically.
                await anyio.wait_writable(self._ssl_sock)
            except ssl.SSLSyscallError as exc:  # pragma: no cover
                # A syscall-level failure inside OpenSSL; the OSError path below is the
                # one a closed socket reproducibly takes.
                raise BrokenResourceError from exc
            except OSError as exc:
                raise BrokenResourceError from exc

    async def send(self, item: bytes) -> None:
        """Encrypt and send all of *item*, awaiting writability as OpenSSL needs it."""
        view = memoryview(item)
        while view:
            sent = await self._retry(self._ssl_sock.send, view)
            view = view[sent:]

    async def receive(self, max_bytes: int = 65536) -> bytes:
        """Receive and decrypt up to *max_bytes*; raise EndOfStream at close_notify/EOF."""
        try:
            data = await self._retry(self._ssl_sock.recv, max_bytes)
        except ssl.SSLEOFError as exc:  # pragma: no cover
            # An abrupt drop with no close_notify; whether OpenSSL surfaces it here or as an
            # empty read below is version dependent, so the empty-read path is the tested one.
            raise EndOfStream from exc
        if not data:
            raise EndOfStream
        return data

    async def send_eof(self) -> None:
        """No-op: TLS has no half-close; the peer learns of the end from aclose."""

    async def aclose(self) -> None:
        """Close the TLS socket, ignoring errors from an already-broken connection."""
        with contextlib.suppress(OSError):
            self._ssl_sock.close()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        """Expose the raw socket, negotiated TLS parameters and the sendfile socket."""
        return {
            SocketAttribute.family: lambda: self._raw_socket.family,
            SocketAttribute.raw_socket: lambda: self._raw_socket,
            SocketAttribute.local_address: self._raw_socket.getsockname,
            SocketAttribute.remote_address: self._raw_socket.getpeername,
            TLSAttribute.alpn_protocol: self._ssl_sock.selected_alpn_protocol,
            TLSAttribute.ssl_object: lambda: self._ssl_sock,
            TLSAttribute.tls_version: self._ssl_sock.version,
            TLSAttribute.peer_certificate: lambda: self._ssl_sock.getpeercert(binary_form=False),
            TLSAttribute.peer_certificate_binary: lambda: self._ssl_sock.getpeercert(
                binary_form=True
            ),
            KTLSAttribute.sendfile_socket: lambda: self._sendfile_socket,
        }


class KTLSAttribute(TypedAttributeSet):
    """Extra typed attributes a KTLSStream exposes beyond the socket and TLS ones."""

    #: The socket to os.sendfile over when kTLS owns the send path, else None.
    sendfile_socket: socket.socket | None = typed_attribute()


class KTLSListener(anyio.abc.Listener[KTLSStream]):  # pragma: linux cover
    """Accepts raw connections and wraps each in socket-backed (kTLS-capable) TLS.

    anyio's ``TLSListener`` wraps an already-accepted stream whose fd the event loop owns,
    so it can only do TLS in userspace. To let the kernel own the send path, the TLS session
    must be bound to the accepted socket itself, so this listener accepts the raw socket and
    hands it straight to :class:`KTLSStream`. As in ``TLSListener``, the handshake runs in
    the per-connection task, so a client that fails it never brings the listener down.
    """

    def __init__(
        self,
        sock: socket.socket,
        ssl_context: ssl.SSLContext,
        handshake_timeout: float | None = None,
    ) -> None:
        self._sock = sock
        self._ssl_context = ssl_context
        self._handshake_timeout = handshake_timeout
        sock.setblocking(False)  # noqa: FBT003

    async def serve(
        self,
        handler: Callable[[KTLSStream], Awaitable[Any]],
        task_group: anyio.abc.TaskGroup | None = None,
    ) -> None:
        """Accept connections forever, running *handler* for each established TLS stream."""
        async with AsyncExitStack() as stack:
            if task_group is None:
                task_group = await stack.enter_async_context(anyio.create_task_group())
            while True:
                conn = await self._accept()
                task_group.start_soon(self._handshake_and_handle, conn, handler)

    async def _accept(self) -> socket.socket:
        while True:
            try:
                conn, _ = self._sock.accept()
            except BlockingIOError:  # noqa: PERF203
                await anyio.wait_readable(self._sock)
            else:
                return conn

    async def _handshake_and_handle(
        self, conn: socket.socket, handler: Callable[[KTLSStream], Awaitable[Any]]
    ) -> None:
        try:
            with anyio.fail_after(self._handshake_timeout):
                stream = await KTLSStream.wrap(conn, ssl_context=self._ssl_context)
        except BaseException as exc:
            conn.close()
            # A single client's bad handshake or timeout must not tear down the listener;
            # only cancellation and other base exceptions are allowed to propagate.
            if not isinstance(exc, anyio.get_cancelled_exc_class()):
                logger.exception("Error during kTLS handshake", exc_info=exc)
            if not isinstance(exc, Exception) or isinstance(exc, anyio.get_cancelled_exc_class()):
                raise
        else:
            await handler(stream)

    async def aclose(self) -> None:
        """Close the listening socket."""
        self._sock.close()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        """Expose the listening socket's family and bound address."""
        return {
            SocketAttribute.family: lambda: self._sock.family,
            SocketAttribute.local_address: self._sock.getsockname,
        }
