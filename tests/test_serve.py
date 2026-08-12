"""Tests for the top-level anycorn.serve entry point."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import anycorn
from anycorn.config import Config


@pytest.mark.anyio
async def test_serve_warns_that_debug_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Debug is a process-setup option, which serve() does not honour, so it warns."""
    monkeypatch.setattr(anycorn, "worker_serve", AsyncMock())
    config = Config()
    config.debug = True
    # AsyncMock reads as a coroutine function, so it wraps as ASGI without being called.
    with pytest.warns(Warning, match="debug"):
        await anycorn.serve(AsyncMock(), config)


@pytest.mark.anyio
async def test_serve_warns_that_workers_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """serve() runs a single worker, so a workers count other than one is ignored with a warning."""
    monkeypatch.setattr(anycorn, "worker_serve", AsyncMock())
    config = Config()
    config.workers = 2
    with pytest.warns(Warning, match="workers"):
        await anycorn.serve(AsyncMock(), config)
