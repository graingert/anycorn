"""Zero-copy file transmission over a socket via ``os.sendfile``.

This backs the ASGI ``http.response.zerocopysend`` extension. anyio's socket
``send`` fully drains to the OS before returning - on asyncio because the stream
protocol sets ``set_write_buffer_limits(0)``, on trio because ``send_all`` writes
everything - so once the response headers have been sent it is safe to hand the
raw socket to ``os.sendfile`` without the file bytes racing ahead of buffered
header bytes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    import socket

# A single sendfile call is capped so a huge file cannot monopolise the socket;
# the loop simply calls again for what is left.
_MAX_SENDFILE_CHUNK = 2**30

#: True when the platform can transmit a file descriptor with a zero-copy syscall.
have_sendfile = hasattr(os, "sendfile")


async def sendfile(sock: socket.socket, in_fd: int, offset: int | None, count: int | None) -> int:
    """Send bytes of ``in_fd`` to ``sock`` with ``os.sendfile``, returning the count sent.

    ``offset`` is where to start reading; ``None`` means the file's current position
    (read once here, since ``os.sendfile`` with an explicit offset does not advance the
    file). ``count`` is how many bytes to send; ``None`` means until end of file.

    The socket is non-blocking (anyio owns it), so ``EAGAIN`` is awaited on writability
    rather than blocking the event loop.
    """
    if offset is None:
        # os.sendfile with an explicit offset never advances the file position, so read
        # the current position and start there - matching "from the current position".
        offset = os.lseek(in_fd, 0, os.SEEK_CUR)

    sock_fd = sock.fileno()
    sent_total = 0
    while count is None or sent_total < count:
        to_send = (
            _MAX_SENDFILE_CHUNK if count is None else min(_MAX_SENDFILE_CHUNK, count - sent_total)
        )
        try:
            sent = os.sendfile(sock_fd, in_fd, offset, to_send)
        except (BlockingIOError, InterruptedError):
            await anyio.wait_writable(sock)
            continue
        if sent == 0:
            break  # End of file reached before count - the app under-declared its length.
        offset += sent
        sent_total += sent
    return sent_total
