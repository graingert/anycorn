"""Tests for the ProtocolWrapper's HTTP/1.1 to HTTP/2 upgrade handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import AsyncMock

import h11
import pytest

import anycorn.protocol
from anycorn.config import Config
from anycorn.events import RawData
from anycorn.protocol import ProtocolWrapper
from anycorn.protocol.h11 import H2CProtocolRequiredError, H2ProtocolAssumedError
from anycorn.typing import ConnectionState
from anycorn.worker_context import WorkerContext

if TYPE_CHECKING:
    from anycorn.events import Event


class _FakeH2Protocol:
    """Stands in for H2Protocol so the upgrade can be observed without a real h2 stack."""

    instances: ClassVar[list[_FakeH2Protocol]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
        self.initiated: list[tuple[tuple, dict]] = []
        self.handled: list[Event] = []
        _FakeH2Protocol.instances.append(self)

    async def initiate(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.initiated.append((args, kwargs))

    async def handle(self, event: Event) -> None:
        self.handled.append(event)


class _RaisesOnHandle:
    """A protocol whose handle raises a given upgrade error, standing in for H11Protocol."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def handle(self, event: Event) -> None:  # noqa: ARG002
        raise self._error


def _wrapper(monkeypatch: pytest.MonkeyPatch, error: Exception) -> ProtocolWrapper:
    monkeypatch.setattr(anycorn.protocol, "H2Protocol", _FakeH2Protocol)
    _FakeH2Protocol.instances.clear()
    wrapper = ProtocolWrapper(
        AsyncMock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        ("127.0.0.1", 1234),
        ("127.0.0.1", 8000),
        AsyncMock(),
        None,
    )
    wrapper.protocol = _RaisesOnHandle(error)  # type: ignore[assignment]
    return wrapper


@pytest.mark.anyio
async def test_alpn_h2_builds_an_h2_protocol_and_initiate_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negotiating h2 over ALPN builds H2Protocol up front, and initiate delegates to it."""
    monkeypatch.setattr(anycorn.protocol, "H2Protocol", _FakeH2Protocol)
    _FakeH2Protocol.instances.clear()
    wrapper = ProtocolWrapper(
        AsyncMock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        None,
        None,
        AsyncMock(),
        None,
        alpn_protocol="h2",
    )
    assert isinstance(wrapper.protocol, _FakeH2Protocol)
    await wrapper.initiate()
    assert wrapper.protocol.initiated == [((), {})]


@pytest.mark.anyio
async def test_h2_preface_upgrades_and_replays_the_buffered_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(monkeypatch, H2ProtocolAssumedError(data=b"leftover"))
    await wrapper.handle(RawData(data=b"PRI * HTTP/2.0"))

    assert isinstance(wrapper.protocol, _FakeH2Protocol)
    assert wrapper.protocol.initiated == [((), {})]  # initiate() with no arguments
    assert wrapper.protocol.handled == [RawData(data=b"leftover")]  # the buffered bytes replayed


@pytest.mark.anyio
async def test_h2_preface_with_no_buffered_data_does_not_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(monkeypatch, H2ProtocolAssumedError(data=b""))
    await wrapper.handle(RawData(data=b"PRI * HTTP/2.0"))

    assert isinstance(wrapper.protocol, _FakeH2Protocol)
    assert wrapper.protocol.handled == []  # nothing buffered, so nothing to replay


@pytest.mark.anyio
async def test_h2c_upgrade_initiates_with_headers_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = h11.Request(
        method="GET",
        target="/",
        headers=[("host", "example"), ("http2-settings", "AAMAAABk"), ("upgrade", "h2c")],
    )
    error = H2CProtocolRequiredError(data=b"rest", request=request)
    wrapper = _wrapper(monkeypatch, error)

    await wrapper.handle(RawData(data=b"GET / HTTP/1.1"))

    assert isinstance(wrapper.protocol, _FakeH2Protocol)
    (args, _kwargs) = wrapper.protocol.initiated[0]
    assert args == (error.headers, error.settings)  # the upgrade carries headers + settings
    assert wrapper.protocol.handled == [RawData(data=b"rest")]


@pytest.mark.anyio
async def test_h2c_upgrade_with_no_buffered_data_does_not_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = h11.Request(method="GET", target="/", headers=[("host", "example")])
    wrapper = _wrapper(monkeypatch, H2CProtocolRequiredError(data=b"", request=request))

    await wrapper.handle(RawData(data=b"GET / HTTP/1.1"))

    assert isinstance(wrapper.protocol, _FakeH2Protocol)
    assert wrapper.protocol.handled == []
