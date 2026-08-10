"""Tests for ProtocolWrapper's HTTP/1.1 to HTTP/2 cleartext (h2c) upgrade.

The ALPN-negotiated and prior-knowledge HTTP/2 paths are driven by a real httpx2
client in ``tests/e2e/test_httpx2.py``. The h2c *Upgrade* mechanism is not something
mainstream clients emit, so it is exercised here against the real H11 and H2 protocols
(only HTTPStream is mocked, so no application task is actually spawned).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

import anycorn.protocol.h2
import anycorn.protocol.h11
from anycorn.config import Config
from anycorn.events import Event, RawData
from anycorn.protocol import ProtocolWrapper
from anycorn.protocol.h2 import H2Protocol
from anycorn.protocol.h11 import H11Protocol
from anycorn.typing import ConnectionState

# A valid HTTP2-Settings header value (base64url of a SETTINGS payload), as an h2c client
# would send, so the real H2 handler accepts the upgrade.
H2C_SETTINGS = b"AAMAAABkAAQAAP__"

# The HTTP/2 connection preface a client sends immediately after the upgrade.
H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"


def _h2c_request(trailing: bytes = b"") -> bytes:
    return (
        b"GET / HTTP/1.1\r\n"
        b"host: anycorn\r\n"
        b"connection: upgrade, http2-settings\r\n"
        b"upgrade: h2c\r\n"
        b"http2-settings: " + H2C_SETTINGS + b"\r\n\r\n" + trailing
    )


def _make_wrapper(monkeypatch: pytest.MonkeyPatch) -> tuple[ProtocolWrapper, list[Event]]:
    # Mock HTTPStream in both protocol modules so the upgrade does not spawn a real app.
    for module in (anycorn.protocol.h11, anycorn.protocol.h2):
        stream_factory = Mock()
        stream_factory.return_value = AsyncMock()
        monkeypatch.setattr(module, "HTTPStream", stream_factory)

    context = Mock()
    context.event_class.return_value = AsyncMock()
    context.mark_request = AsyncMock()
    context.terminate = context.event_class()
    context.terminated = context.event_class()
    context.terminated.is_set.return_value = False

    task_group = Mock()
    task_group.spawn = Mock()  # H2Protocol.initiate spawns its send task synchronously
    task_group.spawn_app = AsyncMock()

    sent: list[Event] = []

    async def send(event: Event) -> None:
        sent.append(event)

    wrapper = ProtocolWrapper(
        AsyncMock(),
        Config(),
        context,
        task_group,
        ConnectionState({}),
        None,
        None,
        send,
        None,
    )
    return wrapper, sent


@pytest.mark.anyio
async def test_h2c_upgrade_switches_to_http2(monkeypatch: pytest.MonkeyPatch) -> None:
    """An h2c Upgrade request replaces the H11 handler with a real H2 handler."""
    wrapper, sent = _make_wrapper(monkeypatch)
    await wrapper.initiate()
    assert isinstance(wrapper.protocol, H11Protocol)

    await wrapper.handle(RawData(data=_h2c_request()))

    assert isinstance(wrapper.protocol, H2Protocol)
    # The 101 Switching Protocols response and the HTTP/2 SETTINGS both went out.
    assert any(isinstance(event, RawData) and b"101" in event.data for event in sent)


@pytest.mark.anyio
async def test_h2c_upgrade_replays_pipelined_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bytes pipelined after the upgrade request are handed to the new HTTP/2 handler."""
    wrapper, _sent = _make_wrapper(monkeypatch)
    await wrapper.initiate()

    # The client pipelines the HTTP/2 preface right behind the upgrade request; it must
    # reach the H2 handler rather than being dropped on the floor with the old H11 one.
    await wrapper.handle(RawData(data=_h2c_request(trailing=H2_PREFACE)))

    assert isinstance(wrapper.protocol, H2Protocol)
