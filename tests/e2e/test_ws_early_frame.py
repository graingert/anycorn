"""Integration test for a websocket frame that arrives with the handshake.

A client that writes its first frame without waiting for the handshake
response puts the frame on the wire in the same breath as the upgrade. On the
server those pipelined bytes are sitting in the connection buffer before the
app has accepted. Whether the app had accepted by the time they were handled
used to come down to the event loop: asyncio ran the app task first and served
the frame, trio did not and answered 400 - so the same client got a different
answer depending on how the server was run.

RFC 6455 forbids a client sending before the handshake completes, so the frame
is answered 400 now, deterministically, before the app is spawned - the answer
no longer depends on the backend.

Unlike the in-memory driver in tests/test_sanity.py, this drives a real,
running anycorn server over a real TCP socket, connected client to server, so
what is exercised is the whole stack down to the kernel's own scheduling of the
two ends - which is the setting the race was actually reported in.

tests/protocol/test_ws_stream.py and tests/protocol/test_h11.py pin the
rejection itself; this adds the end-to-end proof that, over a real socket, a
frame sent ahead of the accept is answered 400 on either backend.
"""

from __future__ import annotations

from http import HTTPStatus
from urllib.parse import urlsplit

import anyio
import pytest
import wsproto
import wsproto.connection
import wsproto.events

import anycorn
from anycorn.config import Config
from tests.helpers import SANITY_BODY, sanity_framework


@pytest.mark.anyio
async def test_frame_sent_with_the_handshake_is_rejected_over_a_real_socket() -> None:
    """A frame in the same write as the upgrade is answered 400, on either backend."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.errorlog = "-"

    async with anyio.create_task_group() as server_tg:
        shutdown = anyio.Event()
        # serve() reports the addresses it actually bound as the task-start value;
        # binding to :0 means the kernel chose the port, so this is how the test
        # learns which one to connect to.
        binds: list[str] = await server_tg.start(
            lambda *, task_status: anycorn.serve(
                sanity_framework, config, shutdown_trigger=shutdown.wait, task_status=task_status
            )
        )
        url = urlsplit(binds[0])
        assert url.hostname is not None
        assert url.port is not None

        stream = await anyio.connect_tcp(url.hostname, url.port)
        async with stream:
            client = wsproto.WSConnection(wsproto.ConnectionType.CLIENT)
            handshake = client.send(wsproto.events.Request(host="anycorn", target="/"))
            # wsproto will not frame a message before it has seen the accept, so the
            # frame is built by the post-handshake connection type - which is what a
            # client that writes both without waiting puts on the wire.
            framer = wsproto.connection.Connection(wsproto.ConnectionType.CLIENT)
            frame = framer.send(wsproto.events.BytesMessage(data=SANITY_BODY))
            # One write, so the upgrade and the frame reach the server together and
            # the frame is handled while WSStream is still mid-handshake.
            await stream.send(handshake + frame)

            events: list[object] = []
            with anyio.fail_after(5):
                while not any(isinstance(e, wsproto.events.RejectConnection) for e in events):
                    data = await stream.receive()
                    assert data != b"", "the connection was closed without an answer"
                    client.receive_data(data)
                    events.extend(client.events())

        shutdown.set()

    reject = events[0]
    assert isinstance(reject, wsproto.events.RejectConnection)
    assert reject.status_code == HTTPStatus.BAD_REQUEST
