"""The interface `H11Protocol` reads and writes a request through.

`H11Protocol` holds one of these and never asks which it has: whether h11 or
httptools is doing the parsing is settled once, in `_make_connection`, and is
invisible from there on. Stating that here rather than as a union of the two
concrete classes means the protocol depends on the interface, and a third parser
would need no change above this line.

Deliberately not implemented by `H11WSConnection`. That passes bytes through
after a websocket upgrade and has no request state to report - it answers None
for the states rather than a `ConnectionState` - so it satisfies the parts of
this the protocol still calls on it, and none of the rest. Making it fit would
mean weakening the contract to whatever both can promise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from . import http1_events as http1


class HTTP1Connection(ABC):
    """A single HTTP/1.1 connection: bytes in, events out, events in, bytes out."""

    # Properties rather than plain annotations, because a bare annotation here is
    # only a declaration: an implementation that never sets the attribute at all
    # is caught by neither ABCMeta nor the type checker, which is no contract.
    # Read-only, since nothing above the connection assigns them - a parser
    # holding its own state keeps it privately and answers through these.

    @property
    @abstractmethod
    def our_state(self) -> http1.ConnectionState:
        """Return how far this side has got through the current response."""

    @property
    @abstractmethod
    def their_state(self) -> http1.ConnectionState:
        """Return how far the peer has got through the current request."""

    @property
    @abstractmethod
    def they_are_waiting_for_100_continue(self) -> bool:
        """Return True while the peer is holding a body back awaiting a 100."""

    @property
    @abstractmethod
    def trailing_data(self) -> tuple[bytes, bool]:
        """Return the bytes read but not consumed, for whoever takes over."""

    @abstractmethod
    def receive_data(self, data: bytes) -> None:
        """Hand *data* to the parser."""

    @abstractmethod
    def next_event(
        self,
    ) -> http1.ReceivableEvent | Literal[http1.Marker.NEED_DATA, http1.Marker.PAUSED]:
        """Return the next event, or a marker when there is not one to return.

        Raises `http1.RemoteProtocolError` on a request that cannot be parsed.
        """

    @abstractmethod
    def send(self, event: http1.SendableEvent) -> bytes:
        """Encode *event* into the bytes to put on the wire.

        Raises `http1.LocalProtocolError` when the event is not one this side may
        send in its current state.
        """

    @abstractmethod
    def start_next_cycle(self) -> None:
        """Ready the connection for the next request on it.

        Raises `http1.LocalProtocolError` unless both sides have finished.
        """
