"""The h11-backed connection, translating to anycorn's HTTP/1.1 vocabulary.

This is the only module that imports h11. Not because h11 might be missing - it
is a hard dependency, and wsproto requires it too - but so that the vocabulary
the protocol speaks belongs to anycorn rather than to whichever parser happens to
be reading the socket.

The translation is thin by design: anycorn's vocabulary was taken from h11's, so
this is mostly a matter of swapping one set of classes for another and mapping
h11's states onto the subset anycorn distinguishes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h11

from . import http1_events as http1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from h11 import Event as H11Event


def _state_map() -> dict[object, http1.ConnectionState]:
    return {
        h11.IDLE: http1.IDLE,
        h11.SEND_RESPONSE: http1.SEND_RESPONSE,
        h11.SEND_BODY: http1.SEND_BODY,
        h11.DONE: http1.DONE,
        h11.MUST_CLOSE: http1.MUST_CLOSE,
        h11.CLOSED: http1.CLOSED,
        h11.ERROR: http1.ERROR,
    }


class H11Connection:
    """anycorn's HTTP/1.1 connection interface, implemented over h11."""

    def __init__(self, max_incomplete_event_size: int) -> None:
        """Start a server-side h11 connection with the configured header limit."""
        self._connection = h11.Connection(
            h11.SERVER, max_incomplete_event_size=max_incomplete_event_size
        )
        self._states = _state_map()

    @property
    def our_state(self) -> http1.ConnectionState:
        """Return our side of the connection, in anycorn's terms."""
        return self._states.get(self._connection.our_state, http1.ERROR)

    @property
    def their_state(self) -> http1.ConnectionState:
        """Return the peer's side of the connection, in anycorn's terms."""
        return self._states.get(self._connection.their_state, http1.ERROR)

    @property
    def they_are_waiting_for_100_continue(self) -> bool:
        """Return True while the peer is holding a body back awaiting a 100."""
        return bool(self._connection.they_are_waiting_for_100_continue)

    @property
    def trailing_data(self) -> tuple[bytes, bool]:
        """Return the bytes read but not consumed, for whoever takes over."""
        return self._connection.trailing_data

    def receive_data(self, data: bytes) -> None:
        """Hand *data* to the parser."""
        self._connection.receive_data(data)

    def next_event(self) -> http1.ReceivableEvent | http1.Marker:
        """Return the next event, translated out of h11's vocabulary."""
        try:
            event = self._connection.next_event()
        except h11.RemoteProtocolError as error:
            raise http1.RemoteProtocolError(
                str(error), error_status_hint=error.error_status_hint
            ) from error

        if event is h11.NEED_DATA:
            return http1.NEED_DATA
        if event is h11.PAUSED:
            return http1.PAUSED
        if isinstance(event, h11.Request):
            return http1.Request(
                method=event.method,
                target=event.target,
                headers=http1.Headers(event.headers.raw_items()),
                http_version=event.http_version,
            )
        if isinstance(event, h11.Data):
            return http1.Data(data=bytes(event.data))
        if isinstance(event, h11.EndOfMessage):
            return http1.EndOfMessage(headers=http1.Headers(event.headers.raw_items()))
        return http1.ConnectionClosed()

    def send(self, event: http1.SendableEvent) -> bytes:
        """Encode *event*, translating into h11's vocabulary to do it."""
        try:
            data = self._connection.send(_to_h11(event))
        except h11.LocalProtocolError as error:
            raise http1.LocalProtocolError(str(error)) from error
        return data if data is not None else b""

    def start_next_cycle(self) -> None:
        """Ready the connection for the next request on it."""
        try:
            self._connection.start_next_cycle()
        except h11.LocalProtocolError as error:
            raise http1.LocalProtocolError(str(error)) from error


def _to_h11(event: http1.SendableEvent) -> H11Event:
    if isinstance(event, http1.Response):
        return h11.Response(status_code=event.status_code, headers=_pairs(event.headers))
    if isinstance(event, http1.InformationalResponse):
        return h11.InformationalResponse(
            status_code=event.status_code, headers=_pairs(event.headers)
        )
    if isinstance(event, http1.Data):
        return h11.Data(data=event.data)
    return h11.EndOfMessage()


def _pairs(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    if isinstance(headers, http1.Headers):
        return headers.raw_items()
    return list(headers)
