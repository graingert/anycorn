"""Tests that spec-conformant ASGI applications satisfy the public ``Framework`` type.

The ASGI spec models the scope and event messages as open-ended dicts, so frameworks
annotate them as mappings rather than as the closed ``TypedDict`` aliases Anycorn uses
internally. These assignments are verified statically by ``ty check`` in CI; the runtime
tests below confirm ``wrap_app`` agrees with the static types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anycorn.app_wrappers import ASGIWrapper
from anycorn.utils import wrap_app

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, MutableMapping

    from anycorn.typing import Framework


class StarletteStyleApp:
    """An app annotated the way Starlette and FastAPI annotate theirs."""

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None: ...


class DictStyleApp:
    """An app annotated with the plain ``dict`` the spec mandates."""

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None: ...


class ReadOnlyScopeApp:
    """An app that never mutates the scope and annotates it read-only."""

    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Awaitable[Mapping[str, Any]]],
        send: Callable[[Mapping[str, Any]], Awaitable[None]],
    ) -> None: ...


class UntypedApp:
    """An app that leaves the ASGI arguments untyped."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None: ...  # noqa: ANN401


async def function_style_app(
    scope: MutableMapping[str, Any],
    receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
    send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
) -> None:
    """Serve as a bare async function rather than a callable class."""


# Static assignability -- these fail ``ty check`` if the boundary type regresses.
_starlette_style: Framework = StarletteStyleApp()
_dict_style: Framework = DictStyleApp()
_read_only_scope: Framework = ReadOnlyScopeApp()
_untyped: Framework = UntypedApp()
_function_style: Framework = function_style_app


def test_spec_conformant_apps_wrap_as_asgi() -> None:
    """Spec-conformant applications are detected and wrapped as ASGI, not WSGI."""
    for app in (
        StarletteStyleApp(),
        DictStyleApp(),
        ReadOnlyScopeApp(),
        UntypedApp(),
        function_style_app,
    ):
        assert isinstance(wrap_app(app, 1024, None), ASGIWrapper)
