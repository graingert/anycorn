"""Tests for the internal stream-level protocol events."""

from __future__ import annotations

import pytest

from anycorn.protocol.events import InformationalResponse


def test_informational_response_accepts_a_1xx_status() -> None:
    event = InformationalResponse(stream_id=1, headers=[], status_code=100)
    assert event.status_code == 100  # noqa: PLR2004


@pytest.mark.parametrize("status_code", [99, 200, 404])
def test_informational_response_rejects_a_non_1xx_status(status_code: int) -> None:
    """A 1XX-only event guards its own invariant so a stray status cannot slip through."""
    with pytest.raises(ValueError, match="Status code must be 1XX"):
        InformationalResponse(stream_id=1, headers=[], status_code=status_code)
