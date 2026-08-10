"""Internal event types used for communication between server components."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


class Event(ABC):  # noqa: B024
    """Base class for all internal server events."""


@dataclass(frozen=True)
class RawData(Event):
    """Event carrying raw bytes received from the network."""

    data: bytes
    address: tuple[str, int] | None = None


@dataclass(frozen=True)
class Closed(Event):
    """Event signalling that a connection has been closed."""


@dataclass(frozen=True)
class Updated(Event):
    """Event signalling a protocol state change."""

    idle: bool


@dataclass(frozen=True)
class SendFile(Event):
    """Event asking the connection to transmit a file descriptor with zero copy.

    Emitted by the HTTP/1.1 protocol for the ``http.response.zerocopysend`` extension:
    the framing bytes go out as ``RawData`` and the body itself as this, so the TCP
    server can ``os.sendfile`` it straight from the file to the socket (or read and
    send it through the TLS stream when the connection is encrypted).
    """

    file: int
    offset: int
    count: int
