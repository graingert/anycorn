"""Tests for the proxy fix middleware."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from anycorn.middleware import ProxyFixMiddleware
from anycorn.middleware.proxy_fix import _get_trusted_value
from anycorn.typing import ConnectionState, HTTPScope

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.anyio
async def test_proxy_fix_legacy() -> None:
    mock = AsyncMock()
    app = ProxyFixMiddleware(mock)
    scope: HTTPScope = {
        "type": "http",
        "asgi": {},
        "http_version": "2",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"x-forwarded-for", b"127.0.0.1"),
            (b"x-forwarded-for", b"127.0.0.2"),
            (b"x-forwarded-proto", b"http,https"),
            (b"x-forwarded-host", b"example.com"),
        ],
        "client": ("127.0.0.3", 80),
        "server": None,
        "extensions": {},
        "state": ConnectionState({}),
    }
    await app(scope, None, None)  # type: ignore[invalid-argument-type]
    mock.assert_called()
    scope = mock.call_args[0][0]
    assert scope["client"] == ("127.0.0.2", 0)
    assert scope["scheme"] == "https"
    host_headers = [h for h in scope["headers"] if h[0].lower() == b"host"]
    assert host_headers == [(b"host", b"example.com")]


@pytest.mark.anyio
async def test_proxy_fix_modern() -> None:
    mock = AsyncMock()
    app = ProxyFixMiddleware(mock, mode="modern")
    scope: HTTPScope = {
        "type": "http",
        "asgi": {},
        "http_version": "2",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"forwarded", b"for=127.0.0.1;proto=http,for=127.0.0.2;proto=https;host=example.com"),
        ],
        "client": ("127.0.0.3", 80),
        "server": None,
        "extensions": {},
        "state": ConnectionState({}),
    }
    await app(scope, None, None)  # type: ignore[invalid-argument-type]
    mock.assert_called()
    scope = mock.call_args[0][0]
    assert scope["client"] == ("127.0.0.2", 0)
    assert scope["scheme"] == "https"
    host_headers = [h for h in scope["headers"] if h[0].lower() == b"host"]
    assert host_headers == [(b"host", b"example.com")]


@pytest.mark.anyio
async def test_proxy_fix_keeps_unpicklable_state() -> None:
    """The scope carries whatever lifespan put in state, which need not be copyable.

    A deepcopy of the whole scope raised TypeError on anything unpicklable - a
    database pool, a client, a lock - so every request through this middleware
    failed. Only client, scheme and headers are rewritten, so a shallow copy does.
    """
    lock = threading.Lock()
    state = {"pool": lock}
    seen: list[Any] = []

    async def app(scope: Any, _receive: Callable, _send: Callable) -> None:  # noqa: ANN401
        seen.append(scope)

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"127.0.0.1")],
        "client": ("localhost", 80),
        "scheme": "http",
        "state": state,
    }

    await ProxyFixMiddleware(app, mode="legacy").__call__(scope, AsyncMock(), AsyncMock())  # type: ignore[arg-type]

    # Handed on, not copied: the app sees the very object lifespan created
    assert seen[0]["state"]["pool"] is lock
    # And the caller's scope was left alone
    assert scope["client"] == ("localhost", 80)
    assert seen[0]["client"] == ("127.0.0.1", 0)


def test_get_trusted_value_with_no_trusted_hops_is_none() -> None:
    """With zero trusted hops nothing in the forwarding headers may be believed."""
    headers = [(b"x-forwarded-for", b"1.2.3.4")]
    assert _get_trusted_value(b"x-forwarded-for", headers, 0) is None


@pytest.mark.anyio
async def test_proxy_fix_passes_non_http_scopes_through_untouched() -> None:
    """Only http and websocket scopes are rewritten; a lifespan scope is forwarded as-is."""
    mock = AsyncMock()
    app = ProxyFixMiddleware(mock)
    scope: Any = {"type": "lifespan", "asgi": {"version": "3.0"}}
    await app(scope, None, None)  # type: ignore[invalid-argument-type]
    assert mock.call_args[0][0] is scope  # the very same object, unmodified


@pytest.mark.anyio
async def test_proxy_fix_modern_ignores_unknown_parts_and_a_missing_for() -> None:
    """A Forwarded value with only proto (and an unrecognised token) sets scheme but no client."""
    mock = AsyncMock()
    app = ProxyFixMiddleware(mock, mode="modern")
    scope: HTTPScope = {
        "type": "http",
        "asgi": {},
        "http_version": "2",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"forwarded", b"proto=https;by=proxy")],
        "client": ("127.0.0.3", 80),
        "server": None,
        "extensions": {},
        "state": ConnectionState({}),
    }
    await app(scope, None, None)  # type: ignore[invalid-argument-type]
    forwarded_scope = mock.call_args[0][0]
    assert forwarded_scope["scheme"] == "https"  # proto was applied
    assert forwarded_scope["client"] == ("127.0.0.3", 80)  # no for=, so client is untouched
