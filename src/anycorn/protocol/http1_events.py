"""anycorn's own vocabulary for the state of an HTTP/1.1 connection.

`H11Protocol` used to speak h11's events and compare against h11's state
sentinels directly, which meant h11 had to be installed even when another parser
was doing the work - and meant the httptools connection had to import h11 purely
to have something to emit.

The names and semantics here are h11's, deliberately: it is the more careful of
the two designs, and matching it keeps the h11 adapter a thin translation rather
than a reinterpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class _Sentinel:
    """A named singleton, so states and markers show their name when printed."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


# Returned by next_event() rather than being events in their own right
NEED_DATA = _Sentinel("NEED_DATA")
PAUSED = _Sentinel("PAUSED")

# Connection states. Only the ones anycorn actually distinguishes are here;
# h11 has a few more that nothing above the connection ever looks at.
IDLE = _Sentinel("IDLE")
SEND_RESPONSE = _Sentinel("SEND_RESPONSE")
SEND_BODY = _Sentinel("SEND_BODY")
DONE = _Sentinel("DONE")
MUST_CLOSE = _Sentinel("MUST_CLOSE")
CLOSED = _Sentinel("CLOSED")
ERROR = _Sentinel("ERROR")


class RemoteProtocolError(Exception):
    """The peer sent something that is not valid HTTP.

    Carries the status to answer with, as h11's does, so the caller does not have
    to work out what a given parse failure deserves.
    """

    def __init__(self, message: str, error_status_hint: int = 400) -> None:
        super().__init__(message)
        self.error_status_hint = error_status_hint


class LocalProtocolError(Exception):
    """We tried to send something that does not fit the connection's state."""


class Headers:
    """Header pairs, readable either normalised or exactly as they arrived.

    Iterating yields lowercased names with the values stripped, which is what
    almost everything wants; `raw_items()` gives back what was on the wire, for
    `h11_pass_raw_headers`.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Iterable[tuple[bytes, bytes]]) -> None:
        self._raw = [(bytes(name), bytes(value)) for name, value in raw]

    def raw_items(self) -> list[tuple[bytes, bytes]]:
        """Return the pairs with the case and spacing they were sent with."""
        return list(self._raw)

    def __iter__(self) -> Iterator[tuple[bytes, bytes]]:
        for name, value in self._raw:
            yield name.lower(), value.strip()

    def __len__(self) -> int:
        return len(self._raw)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Headers):
            return list(self) == list(other)
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return f"Headers({self._raw!r})"


@dataclass(frozen=True)
class Request:
    """A request head, as read off the connection."""

    method: bytes
    target: bytes
    headers: Headers
    http_version: bytes


@dataclass(frozen=True)
class Data:
    """A piece of a body, in either direction."""

    data: bytes


@dataclass(frozen=True)
class EndOfMessage:
    """The end of a request or response body."""

    headers: Headers = field(default_factory=lambda: Headers([]))


@dataclass(frozen=True)
class ConnectionClosed:
    """The peer will send nothing further."""


@dataclass(frozen=True)
class Response:
    """A final response head, to send."""

    status_code: int
    headers: Iterable[tuple[bytes, bytes]]


@dataclass(frozen=True)
class InformationalResponse:
    """A 1xx response head, to send."""

    status_code: int
    headers: Iterable[tuple[bytes, bytes]]


SendableEvent = Response | InformationalResponse | Data | EndOfMessage
