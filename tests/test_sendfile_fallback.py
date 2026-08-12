"""Tests for the read-based fallbacks in sendfile, which run on every platform.

Unlike ``test_sendfile``/``test_zerocopysend``, these do not need ``os.sendfile``: they
cover ``read_file_chunks`` (the path used for TLS, HTTP/2 and HTTP/3, or where sendfile is
missing) and ``_pread``, including the Windows-style dup/seek/read fallback exercised here
on any platform by hiding ``os.pread``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from anycorn.sendfile import _pread, read_file_chunks

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.anyio
async def test_read_file_chunks_yields_the_whole_window(tmp_path: Path) -> None:
    """The requested window is yielded in order across chunk boundaries."""
    payload = bytes(range(256)) * 600  # ~150 KiB, several 64 KiB chunks
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    with file_path.open("rb") as file:
        collected = b"".join([chunk async for chunk in read_file_chunks(file.fileno(), 10, 5000)])
    assert collected == payload[10:5010]


@pytest.mark.anyio
async def test_read_file_chunks_stops_at_end_of_file(tmp_path: Path) -> None:
    """A count past the end of the file stops cleanly at EOF rather than looping."""
    payload = b"short"
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    with file_path.open("rb") as file:
        collected = b"".join(
            [chunk async for chunk in read_file_chunks(file.fileno(), 0, 1_000_000)]
        )
    assert collected == payload  # only what the file actually held


def test_pread_reads_at_offset_without_moving_the_position(tmp_path: Path) -> None:
    """_pread returns bytes at the offset and leaves the file position untouched."""
    payload = b"0123456789"
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    with file_path.open("rb") as file:
        fd = file.fileno()
        os.lseek(fd, 2, os.SEEK_SET)
        assert _pread(fd, 3, 5) == b"567"
        if hasattr(os, "pread"):  # pragma: no branch - constant per platform
            # os.pread reads without moving the position; the Windows dup/seek fallback
            # shares the descriptor's offset, so it does move there and is not asserted.
            assert os.lseek(fd, 0, os.SEEK_CUR) == 2  # noqa: PLR2004  # pragma: win32 no cover


def test_pread_falls_back_to_dup_when_pread_is_absent(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Where os.pread is unavailable (Windows), a duplicated descriptor is seeked and read.

    os.pread is hidden here so the fallback is exercised on any platform, and it returns
    the bytes at the offset. (It reads through a duplicated descriptor, which shares the
    original's offset, so the caller's position does move - only the returned bytes are
    asserted.)
    """
    monkeypatch.delattr(os, "pread", raising=False)
    payload = b"abcdefghij"
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(payload)
    with file_path.open("rb") as file:
        assert _pread(file.fileno(), 4, 3) == b"defg"
