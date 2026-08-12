"""Tests for the ASGI task handler's error handling in the task group."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, cast

import anyio
import anyio.from_thread
import anyio.to_thread
import pytest

from anycorn.config import Config
from anycorn.task_group import _handle

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    _ExceptionGroup = BaseExceptionGroup  # noqa: F821 - builtin on 3.11+
else:  # pragma: <3.11 cover
    from exceptiongroup import BaseExceptionGroup as _ExceptionGroup

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anycorn.typing import Scope


async def _run_handle(app: Callable[..., Awaitable[None]]) -> list[object]:
    """Drive _handle with *app* over throwaway channels, collecting what it sends."""
    sent: list[object] = []

    async def send(message: object) -> None:
        sent.append(message)

    channels = anyio.create_memory_object_stream[Any](10)
    await _handle(
        cast("Any", app),
        Config(),
        cast("Scope", {"type": "http"}),
        channels,
        send,
        anyio.to_thread.run_sync,
        anyio.from_thread.run,
    )
    return sent


@pytest.mark.anyio
async def test_handle_reports_an_application_error_group() -> None:
    """A group carrying a real application error is logged and the stream is ended."""

    async def app(*_args: Any) -> None:  # noqa: ANN401
        raise _ExceptionGroup("app failed", [ValueError("boom")])

    sent = await _run_handle(app)
    # One None from the except branch, one from the finally - both end the stream.
    assert sent == [None, None]


@pytest.mark.anyio
async def test_handle_reports_a_plain_application_error() -> None:
    """A plain (ungrouped) exception is logged, and the finally still ends the stream."""

    async def app(*_args: Any) -> None:  # noqa: ANN401
        raise ValueError("boom")

    sent = await _run_handle(app)
    assert sent == [None]  # only the finally ends the stream on the plain-exception path
