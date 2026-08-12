"""Tests for the UDP (QUIC/HTTP-3) server's loop and send handling."""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import anycorn.protocol.quic
from anycorn.config import Config
from anycorn.events import Closed, RawData
from anycorn.udp_server import UDPServer
from anycorn.worker_context import WorkerContext


@pytest.mark.anyio
async def test_run_exits_without_reading_when_already_terminated_and_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that is terminated with an idle protocol closes out without a read."""

    class _IdleQuic:
        def __init__(self, *_args: Any) -> None:  # noqa: ANN401
            self.idle = True
            self.close_all = AsyncMock()

    monkeypatch.setattr(anycorn.protocol.quic, "QuicProtocol", _IdleQuic)

    udp_socket = MagicMock()
    udp_socket.socket.family = socket.AF_INET
    udp_socket.socket.getsockname.return_value = ("127.0.0.1", 4433)

    context = WorkerContext(None)
    await context.terminated.set()  # already terminated, so the read loop must not run

    server = UDPServer(AsyncMock(), Config(), context, {}, udp_socket)
    await server.run()

    udp_socket.receive.assert_not_called()  # the loop condition was false from the start
    server.protocol.close_all.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_protocol_send_ignores_non_rawdata_events() -> None:
    """Only RawData is written to the socket; other protocol events are not."""
    udp_socket = AsyncMock()
    server = UDPServer(AsyncMock(), Config(), WorkerContext(None), {}, udp_socket)

    await server.protocol_send(Closed())
    udp_socket.sendto.assert_not_called()


@pytest.mark.anyio
async def test_protocol_send_writes_rawdata_to_its_address() -> None:
    """A RawData event is sent to the address it names."""
    udp_socket = AsyncMock()
    server = UDPServer(AsyncMock(), Config(), WorkerContext(None), {}, udp_socket)

    await server.protocol_send(RawData(data=b"quic", address=("127.0.0.1", 4433)))
    udp_socket.sendto.assert_awaited_once_with(b"quic", "127.0.0.1", 4433)
