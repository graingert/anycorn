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
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class ConnectionState(Enum):
    """How much of a request or response each side has got through.

    Only the states anycorn actually distinguishes are here; h11 has several more
    that nothing above the connection ever looks at.
    """

    IDLE = auto()
    SEND_RESPONSE = auto()
    SEND_BODY = auto()
    DONE = auto()
    MUST_CLOSE = auto()
    CLOSED = auto()
    ERROR = auto()


class Marker(Enum):
    """Returned by `next_event()` in place of an event."""

    NEED_DATA = auto()
    PAUSED = auto()


# Bound at module level so call sites read as they did against h11 - `http1.DONE`
# rather than `http1.ConnectionState.DONE` - while still being properly typed
IDLE = ConnectionState.IDLE
SEND_RESPONSE = ConnectionState.SEND_RESPONSE
SEND_BODY = ConnectionState.SEND_BODY
DONE = ConnectionState.DONE
MUST_CLOSE = ConnectionState.MUST_CLOSE
CLOSED = ConnectionState.CLOSED
ERROR = ConnectionState.ERROR

NEED_DATA = Marker.NEED_DATA
PAUSED = Marker.PAUSED


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


ReceivableEvent = Request | Data | EndOfMessage | ConnectionClosed
SendableEvent = Response | InformationalResponse | Data | EndOfMessage
