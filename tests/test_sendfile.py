"""Tests for the zero-copy sendfile helper, driven over a real loopback TCP connection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import pytest

from anycorn.sendfile import have_sendfile, sendfile

from .helpers import tcp_socket_pair

if TYPE_CHECKING:
    import socket
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

pytestmark = pytest.mark.skipif(not have_sendfile, reason="os.sendfile unavailable")


async def _drain(sock: socket.socket, expected: int) -> bytes:
    received = bytearray()
    while len(received) < expected:
        chunk = await anyio.to_thread.run_sync(sock.recv, 65536)
        if not chunk:
            break
        received.extend(chunk)
    return bytes(received)


@pytest.mark.anyio
async def test_sendfile_transmits_the_whole_file(tmp_path: Path) -> None:
    """A file larger than the socket buffer is sent in full, awaiting writability."""
    payload = b"zero-copy payload\n" * 100_000  # ~1.8 MiB, well over the socket buffer
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)
    file = await anyio.Path(file_path).open("rb")
    in_fd = file.wrapped.fileno()
    send_sock, recv_sock = tcp_socket_pair()
    send_sock.setblocking(False)  # noqa: FBT003

    received = b""
    try:
        async with anyio.create_task_group() as task_group:

            async def reader() -> None:
                nonlocal received
                received = await _drain(recv_sock, len(payload))

            task_group.start_soon(reader)
            sent = await sendfile(send_sock, in_fd, 0, len(payload))
            assert sent == len(payload)
    finally:
        await file.aclose()
        send_sock.close()
        recv_sock.close()

    assert received == payload


@pytest.mark.anyio
async def test_sendfile_honours_offset_and_count(tmp_path: Path) -> None:
    """Only the requested window of the file is sent."""
    payload = bytes(range(256)) * 8
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)
    file = await anyio.Path(file_path).open("rb")
    in_fd = file.wrapped.fileno()
    send_sock, recv_sock = tcp_socket_pair()
    send_sock.setblocking(False)  # noqa: FBT003

    received = b""
    try:
        async with anyio.create_task_group() as task_group:

            async def reader() -> None:
                nonlocal received
                received = await _drain(recv_sock, 100)

            task_group.start_soon(reader)
            sent = await sendfile(send_sock, in_fd, 10, 100)
            assert sent == 100  # noqa: PLR2004
    finally:
        await file.aclose()
        send_sock.close()
        recv_sock.close()

    assert received == payload[10:110]


@pytest.mark.anyio
async def test_sendfile_without_count_reads_to_eof(tmp_path: Path) -> None:
    """count=None sends from the offset to the end of the file."""
    payload = b"tail" * 1000
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)
    file = await anyio.Path(file_path).open("rb")
    in_fd = file.wrapped.fileno()
    send_sock, recv_sock = tcp_socket_pair()
    send_sock.setblocking(False)  # noqa: FBT003

    received = b""
    try:
        async with anyio.create_task_group() as task_group:

            async def reader() -> None:
                nonlocal received
                received = await _drain(recv_sock, len(payload) - 4)

            task_group.start_soon(reader)
            sent = await sendfile(send_sock, in_fd, 4, None)
            assert sent == len(payload) - 4
    finally:
        await file.aclose()
        send_sock.close()
        recv_sock.close()

    assert received == payload[4:]


@pytest.mark.anyio
async def test_sendfile_waits_for_writability_on_eagain(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A would-block from os.sendfile is awaited on writability, then the send resumes.

    EAGAIN is provoked deterministically by making the first os.sendfile raise
    BlockingIOError; the socket is really writable, so wait_writable returns at once and
    the retry sends the payload in full.
    """
    payload = b"retry after eagain\n" * 10
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)
    file = await anyio.Path(file_path).open("rb")
    in_fd = file.wrapped.fileno()
    send_sock, recv_sock = tcp_socket_pair()
    send_sock.setblocking(False)  # noqa: FBT003

    real_sendfile = os.sendfile
    calls = 0

    def flaky_sendfile(*args: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlockingIOError  # force the EAGAIN branch once
        return real_sendfile(*args)

    monkeypatch.setattr(os, "sendfile", flaky_sendfile)

    received = b""
    try:
        async with anyio.create_task_group() as task_group:

            async def reader() -> None:
                nonlocal received
                received = await _drain(recv_sock, len(payload))

            task_group.start_soon(reader)
            sent = await sendfile(send_sock, in_fd, 0, len(payload))
            assert sent == len(payload)
    finally:
        await file.aclose()
        send_sock.close()
        recv_sock.close()

    assert calls >= 2  # noqa: PLR2004  # the first call blocked, a later one sent
    assert received == payload


@pytest.mark.anyio
async def test_sendfile_offset_none_uses_current_position(tmp_path: Path) -> None:
    """offset=None starts from the file's current position."""
    payload = b"0123456789"
    file_path = tmp_path / "payload.bin"
    await anyio.Path(file_path).write_bytes(payload)
    file = await anyio.Path(file_path).open("rb")
    in_fd = file.wrapped.fileno()
    os.lseek(in_fd, 3, os.SEEK_SET)
    send_sock, recv_sock = tcp_socket_pair()
    send_sock.setblocking(False)  # noqa: FBT003

    received = b""
    try:
        async with anyio.create_task_group() as task_group:

            async def reader() -> None:
                nonlocal received
                received = await _drain(recv_sock, 7)

            task_group.start_soon(reader)
            sent = await sendfile(send_sock, in_fd, None, None)
            assert sent == 7  # noqa: PLR2004
    finally:
        await file.aclose()
        send_sock.close()
        recv_sock.close()

    assert received == payload[3:]
