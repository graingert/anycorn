"""Tests for worker startup and socket lifecycle."""

from __future__ import annotations

import signal
import socket
import sys
from functools import partial
from pickle import PicklingError
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import anyio
import anyio.abc
import anyio.to_thread
import pytest

import anycorn.run
from anycorn.config import Config
from anycorn.datagram import wrap_datagram_socket
from anycorn.events import RawData
from anycorn.run import run, worker_serve
from anycorn.udp_server import UDPServer
from anycorn.utils import load_application, wrap_app
from anycorn.worker_context import WorkerContext

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from tests.conftest import TLSCerts

if sys.version_info < (3, 11):  # pragma: <3.11 cover
    from exceptiongroup import BaseExceptionGroup


async def app(scope: Any, _receive: Any, send: Any) -> None:  # noqa: ANN401
    assert scope["type"] == "http"
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


class _FakeSignalModule:
    """Stands in for the real ``signal`` module, scoped to run.py's own namespace.

    Any test that drives run() into its multiprocess branch needs this: the very
    first thing that branch does, before calling _populate or anything else, is
    a real ``signal.signal(SIGINT, SIG_IGN)``. Replacing the real, process-wide
    module's ``.signal`` attribute would intercept calls from unrelated code
    running during the test too, and a signal handler actually installed on the
    real module would outlive monkeypatch's undo (it only reverts attribute
    assignments, not OS-level signal state). Rebinding the ``signal`` name
    inside anycorn.run's own namespace avoids both: the real module, and every
    other test, are never touched.
    """

    SIGINT = signal.SIGINT
    SIGTERM = signal.SIGTERM
    SIG_IGN = signal.SIG_IGN
    if hasattr(signal, "SIGHUP"):  # pragma: no branch - constant per platform
        SIGHUP = signal.SIGHUP  # pragma: win32 no cover

    def __init__(self) -> None:
        self.calls: list[tuple[int, Callable]] = []

    def signal(self, signalnum: int, handler: Callable) -> None:
        self.calls.append((signalnum, handler))


@pytest.mark.anyio
async def test_worker_serve_closes_quic_sockets(tls_certs: TLSCerts) -> None:
    """QUIC sockets are opened by create_sockets(), so the worker must close them."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.quic_bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)

    sockets = config.create_sockets()
    quic_sockets = list(sockets.quic_sockets)
    assert quic_sockets, "expected create_sockets to bind a QUIC socket"
    for sock in sockets.secure_sockets:
        sock.listen(config.backlog)

    shutdown = anyio.Event()
    async with anyio.create_task_group() as tg:
        binds = await tg.start(
            partial(
                worker_serve,
                wrap_app(app, config.wsgi_max_body_size, None),
                config,
                sockets=sockets,
                shutdown_trigger=shutdown.wait,
            )
        )
        assert len(binds) == len(sockets.secure_sockets) + len(quic_sockets)
        shutdown.set()

    assert [sock.fileno() for sock in quic_sockets] == [-1] * len(quic_sockets)


@pytest.mark.anyio
async def test_worker_serve_marks_terminated_before_the_servers_unwind(
    tls_certs: TLSCerts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server must be able to see the worker is going down whilst it still runs.

    QuicProtocol refuses a new connection whilst terminated is set, so marking it only
    once the servers have been cancelled would leave that check unable to refuse
    anything - a QUIC Initial arriving as shutdown starts would be taken on by a worker
    that is already going away.
    """
    observed: list[bool] = []

    class _RecordingUDPServer(UDPServer):
        async def run(
            self, *, task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED
        ) -> None:
            try:
                await super().run(task_status=task_status)
            finally:
                observed.append(self.context.terminated.is_set())

    monkeypatch.setattr(anycorn.run, "UDPServer", _RecordingUDPServer)

    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.quic_bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)

    shutdown = anyio.Event()
    async with anyio.create_task_group() as tg:
        await tg.start(
            partial(
                worker_serve,
                wrap_app(app, config.wsgi_max_body_size, None),
                config,
                shutdown_trigger=shutdown.wait,
            )
        )
        shutdown.set()

    assert observed == [True]


@pytest.mark.anyio
async def test_udp_server_serialises_concurrent_sends() -> None:
    """QUIC sends from several tasks at once must not collide on the socket.

    anyio guards a socket against concurrent writers rather than interleaving them,
    so the timer and stream tasks sending alongside the read loop would otherwise
    raise `BusyResourceError` at whichever one lost the race.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)  # noqa: FBT003
    datagram_socket = await wrap_datagram_socket(sock)
    server = UDPServer(AsyncMock(), Config(), WorkerContext(None), {}, datagram_socket)

    try:
        async with anyio.create_task_group() as task_group:
            for _ in range(20):
                task_group.start_soon(
                    server.protocol_send,
                    RawData(data=b"x" * 1024, address=("127.0.0.1", 9999)),
                )
    finally:
        await datagram_socket.aclose()


def test_run_closes_sockets_when_it_raises(
    tls_certs: TLSCerts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's sockets are closed however run() exits, not only when it finishes."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.quic_bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    sockets = config.create_sockets()
    monkeypatch.setattr(config, "create_sockets", lambda: sockets)

    # The earliest thing run() refuses to do, so it raises with the sockets already open
    config.use_reloader = True
    config.workers = 0
    with pytest.raises(RuntimeError, match="Cannot reload without workers"):
        run(config)

    opened = [*sockets.secure_sockets, *sockets.insecure_sockets, *sockets.quic_sockets]
    assert opened, "expected create_sockets to bind something"
    assert [sock.fileno() for sock in opened] == [-1] * len(opened)


def test_populate_sets_process_daemon_from_config() -> None:
    """Each spawned worker process must pick up config.daemon, not a hardcoded value."""

    class _Process:
        def __init__(self, target: Any, kwargs: dict) -> None:  # noqa: ANN401
            self.target = target
            self.kwargs = kwargs
            self.daemon = False

        def start(self) -> None:
            pass

    class _Ctx:
        def Process(self, *, target: Any, kwargs: dict) -> _Process:  # noqa: ANN401, N802
            return _Process(target, kwargs)

    config = Config()
    config.daemon = False
    processes: list = []
    anycorn.run._populate(processes, config, lambda **_kwargs: None, None, None, _Ctx())  # type: ignore[arg-type]
    assert processes[0].daemon is False


@pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP does not exist on Windows.")
def test_run_registers_sighup_to_reload_workers(  # pragma: win32 no cover
    tls_certs: TLSCerts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGHUP must be wired up to gracefully restart workers, not left to the default action."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    sockets = config.create_sockets()
    monkeypatch.setattr(config, "create_sockets", lambda: sockets)
    config.workers = 1

    class _Process:
        def __init__(self) -> None:
            self.sentinel = object()
            self.exitcode = 1  # non-zero, so run() stops after a single pass

        def join(self) -> None:
            pass

    def _populate(processes: list, *_args: object, **_kwargs: object) -> None:
        processes.append(_Process())

    fake_signal = _FakeSignalModule()

    monkeypatch.setattr(anycorn.run, "_populate", _populate)
    monkeypatch.setattr(anycorn.run, "wait", lambda _sentinels: None)
    monkeypatch.setattr(anycorn.run, "signal", fake_signal)

    run(config)

    assert signal.signal is not fake_signal.signal  # the real module was never touched

    sighup_handlers = [
        handler for signalnum, handler in fake_signal.calls if signalnum == signal.SIGHUP
    ]
    assert len(sighup_handlers) == 1
    assert getattr(sighup_handlers[0], "__name__", None) == "reload"


def test_run_terminates_workers_when_it_raises(
    tls_certs: TLSCerts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that fails partway signals its workers rather than orphaning them."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    sockets = config.create_sockets()
    monkeypatch.setattr(config, "create_sockets", lambda: sockets)
    config.workers = 2

    terminated: list[object] = []

    class _Process:
        sentinel = None
        exitcode = None

        def terminate(self) -> None:
            terminated.append(self)

    def _spawn_then_fail(processes: list, *_args: object, **_kwargs: object) -> None:
        # One worker up, the next refusing - the shape that used to walk past the
        # terminate loop entirely
        processes.append(_Process())
        raise PicklingError

    fake_signal = _FakeSignalModule()

    monkeypatch.setattr(anycorn.run, "_populate", _spawn_then_fail)
    monkeypatch.setattr(anycorn.run, "signal", fake_signal)

    with pytest.raises(PicklingError):
        run(config)

    assert signal.signal is not fake_signal.signal  # the real module was never touched
    assert len(terminated) == 1


import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

_RUN_APP = "tests/assets/run_app.py:app"


@pytest.fixture
def _restore_signals() -> Iterator[None]:
    """Save and restore the process-wide signal handlers run() installs.

    run() does not put back the handlers it sets, so a test driving it for real would
    otherwise leave stale handlers behind for whatever pytest runs next.
    """
    names = [n for n in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK") if hasattr(signal, n)]
    saved = {n: signal.getsignal(getattr(signal, n)) for n in names}
    try:
        yield
    finally:
        for name, handler in saved.items():
            signal.signal(getattr(signal, name), handler)


def _wait_for_handler(signum: int, baseline: object) -> None:  # pragma: win32 no cover
    """Block until run()/anyio installs a real handler for *signum* (or a timeout)."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:  # pragma: no branch - handler always arrives
        if signal.getsignal(signum) is not baseline:
            return
        time.sleep(0.02)


def _deliver_signal_once_installed(signum: int, baseline: object) -> None:  # pragma: win32 no cover
    """From a helper thread, wait until a real handler for *signum* is in place, then send it.

    Sending only after the handler is installed avoids the window in which the default
    action (terminate) would otherwise take the test process down.
    """
    _wait_for_handler(signum, baseline)
    os.kill(os.getpid(), signum)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery to self.")
@pytest.mark.usefixtures("_restore_signals")
def test_run_workers_zero_serves_in_process_until_signalled(  # pragma: win32 no cover
    tmp_path: Path,
) -> None:
    """With no workers, run() serves in-process and shuts down on a signal, writing the pid."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.workers = 0
    config.application_path = _RUN_APP
    config.pid_path = str(tmp_path / "anycorn.pid")

    baseline = signal.getsignal(signal.SIGTERM)
    thread = threading.Thread(
        target=_deliver_signal_once_installed, args=(signal.SIGTERM, baseline)
    )
    thread.start()
    try:
        assert run(config) == 0
    finally:
        thread.join()
    assert (tmp_path / "anycorn.pid").read_text() == str(os.getpid())


@pytest.mark.usefixtures("_restore_signals")
def test_run_returns_nonzero_when_a_worker_fails_to_start() -> None:
    """A worker whose application cannot be imported exits non-zero, and run() reports it."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.workers = 1
    config.application_path = "anycorn_no_such_application_module:app"  # fails to import

    assert run(config) != 0  # the failing worker was spawned, reaped and reported


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery to self.")
@pytest.mark.usefixtures("_restore_signals")
def test_run_multiprocess_shuts_down_gracefully_on_sigterm() -> None:  # pragma: win32 no cover
    """A SIGTERM to a running multi-worker server stops it cleanly with a zero exit."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.workers = 1
    config.application_path = _RUN_APP

    baseline = signal.getsignal(signal.SIGTERM)
    thread = threading.Thread(
        target=_deliver_signal_once_installed, args=(signal.SIGTERM, baseline)
    )
    thread.start()
    try:
        assert run(config) == 0
    finally:
        thread.join()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery to self.")
@pytest.mark.usefixtures("_restore_signals")
def test_run_reloader_reloads_on_a_file_change(tmp_path: Path) -> None:  # pragma: win32 no cover
    """A change to a watched file reloads the workers before a SIGTERM ends the run."""
    app_file = tmp_path / "reload_app.py"
    app_file.write_text(
        "async def app(scope, receive, send):\n"
        "    await send({'type': 'http.response.start', 'status': 200, 'headers': []})\n"
        "    await send({'type': 'http.response.body', 'body': b''})\n"
    )
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.workers = 1
    config.use_reloader = True
    config.application_path = f"{app_file}:app"

    hup_baseline = signal.getsignal(signal.SIGHUP)

    def _change_then_stop() -> None:
        # Once the reloader loop is running (its SIGHUP handler is in place), touch the
        # watched file so check_for_updates fires a reload, then stop the run. The pause
        # lets files_to_watch record the original mtime before it is bumped.
        _wait_for_handler(signal.SIGHUP, hup_baseline)
        time.sleep(0.5)
        future = time.time() + 30
        os.utime(app_file, (future, future))  # a newer mtime the reloader will notice
        time.sleep(2)  # let the reload happen and the loop respawn
        os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=_change_then_stop)
    thread.start()
    try:
        run(config)
    finally:
        thread.join()


def test_anyio_worker_runs_worker_serve_to_shutdown() -> None:
    """anyio_worker really loads the app and runs worker_serve, which a set event ends.

    A plain sync test: anyio_worker blocks on its own anyio.run, so it is called directly
    rather than offloaded to a worker thread (which finalised sockets flakily under trio).
    """
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.application_path = _RUN_APP
    sockets = config.create_sockets()

    event = anycorn.run.get_context("spawn").Event()
    event.set()  # already set, so worker_serve shuts down as soon as it starts

    anycorn.run.anyio_worker(config, sockets, event)


@pytest.mark.anyio
async def test_worker_serve_answers_a_request_for_a_path_loaded_app() -> None:
    """A path-loaded app answers a full GET through worker_serve, exercising its whole response.

    Runs worker_serve directly in the test's own event loop rather than through an
    anyio_worker worker thread, so it is deterministic on trio as well as asyncio while still
    covering the load-by-path app end to end.
    """
    import httpx2  # noqa: PLC0415

    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.accesslog = "-"
    config.errorlog = "-"
    app = load_application(_RUN_APP, config.wsgi_max_body_size)

    shutdown = anyio.Event()
    with anyio.fail_after(10):
        async with anyio.create_task_group() as task_group:
            binds = await task_group.start(
                partial(worker_serve, app, config, shutdown_trigger=shutdown.wait)
            )
            async with httpx2.AsyncClient(base_url=binds[0]) as client:
                response = await client.get("/")
            shutdown.set()

    assert response.status_code == 200  # noqa: PLR2004
    """A worker with no exitcode yet is not reaped; a finished one is."""

    class _Process:
        def __init__(self, exitcode: int | None) -> None:
            self.exitcode = exitcode

        def join(self) -> None:
            pass

    running, finished = _Process(None), _Process(0)
    processes: list = [running, finished]
    anycorn.run._join_exited(processes)
    assert processes == [running]


def test_populate_raises_a_clear_error_when_the_config_cannot_be_pickled() -> None:
    """A PicklingError from process.start becomes a helpful RuntimeError."""

    class _Process:
        daemon = False

        def start(self) -> None:
            raise PicklingError

    class _Ctx:
        def Process(self, *, target: Any, kwargs: dict) -> _Process:  # noqa: ANN401, ARG002, N802
            return _Process()

    with pytest.raises(RuntimeError, match="Cannot pickle the config"):
        anycorn.run._populate([], Config(), lambda **_k: None, None, None, _Ctx())  # type: ignore[arg-type]


def test_anyio_worker_serves_tls_with_a_request_limit(tls_certs: TLSCerts) -> None:
    """The worker listens on the secure socket, applies max_requests, and a set event ends it."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.application_path = _RUN_APP
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    config.max_requests = 100  # exercises the jitter line
    sockets = config.create_sockets()
    event = anycorn.run.get_context("spawn").Event()
    event.set()
    anycorn.run.anyio_worker(config, sockets, event)


def test_anyio_worker_without_sockets_creates_its_own() -> None:
    """With no sockets passed, the worker binds its own before serving.

    anyio_worker is a blocking call that runs its own anyio.run internally, so this is a
    plain sync test with no outer event loop to offload it from - which also avoids the
    flaky socket finalisation seen when it was driven through an anyio worker thread.
    """
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.application_path = _RUN_APP
    event = anycorn.run.get_context("spawn").Event()
    event.set()
    anycorn.run.anyio_worker(config, None, event)


@pytest.mark.anyio
async def test_worker_serve_uses_ktls_when_available(
    tls_certs: TLSCerts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With kTLS reported available and requested, the worker builds a KTLSListener.

    kTLS activation itself needs a Linux kernel and a kTLS-capable OpenSSL, which cannot
    be conjured here, so only the capability flag is forced; the listener is still built
    and served for real.
    """
    monkeypatch.setattr(anycorn.run, "can_enable_ktls", True)
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    config.use_ktls = True
    sockets = config.create_sockets()
    for sock in sockets.secure_sockets:
        sock.listen(config.backlog)

    shutdown = anyio.Event()
    async with anyio.create_task_group() as tg:
        binds = await tg.start(
            partial(
                worker_serve,
                wrap_app(app, config.wsgi_max_body_size, None),
                config,
                sockets=sockets,
                shutdown_trigger=shutdown.wait,
            )
        )
        assert binds
        assert binds[0].startswith("https://")
        shutdown.set()


@pytest.mark.anyio
async def test_worker_serve_warns_when_ktls_requested_but_unavailable(
    tls_certs: TLSCerts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_ktls set on a host without kTLS logs a warning and serves ordinary userspace TLS.

    Forcing the flag off runs this path even where kTLS is real, so the warning branch is
    covered on every job rather than only the ones that happen to lack kTLS.
    """
    monkeypatch.setattr(anycorn.run, "can_enable_ktls", False)
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.certfile = str(tls_certs.certfile)
    config.keyfile = str(tls_certs.keyfile)
    config.use_ktls = True
    sockets = config.create_sockets()
    for sock in sockets.secure_sockets:
        sock.listen(config.backlog)

    shutdown = anyio.Event()
    async with anyio.create_task_group() as tg:
        binds = await tg.start(
            partial(
                worker_serve,
                wrap_app(app, config.wsgi_max_body_size, None),
                config,
                sockets=sockets,
                shutdown_trigger=shutdown.wait,
            )
        )
        assert binds
        assert binds[0].startswith("https://")
        shutdown.set()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["trio"])
async def test_worker_serve_without_a_trigger_on_trio(
    anyio_backend: str,  # noqa: ARG001
) -> None:
    """On trio with no shutdown trigger, the server runs until its group is cancelled."""
    config = Config()
    config.bind = ["127.0.0.1:0"]

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            await tg.start(
                partial(worker_serve, wrap_app(app, config.wsgi_max_body_size, None), config)
            )
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_worker_serve_reraises_a_non_shutdown_error() -> None:
    """An error that is not a shutdown propagates out of the server rather than being eaten."""
    config = Config()
    config.bind = ["127.0.0.1:0"]

    async def _boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await worker_serve(
            wrap_app(app, config.wsgi_max_body_size, None), config, shutdown_trigger=_boom
        )
    assert exc_info.value.subgroup(RuntimeError) is not None


@pytest.mark.anyio
async def test_worker_serve_answers_a_real_request() -> None:
    """A plaintext worker serves a real GET, exercising the app end to end."""
    import httpx2  # noqa: PLC0415

    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.accesslog = "-"
    config.errorlog = "-"
    shutdown = anyio.Event()
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            binds = await tg.start(
                partial(
                    worker_serve,
                    wrap_app(app, config.wsgi_max_body_size, None),
                    config,
                    shutdown_trigger=shutdown.wait,
                )
            )
            async with httpx2.AsyncClient(base_url=binds[0]) as client:
                response = await client.get("/")
            shutdown.set()
    assert response.status_code == 200  # noqa: PLR2004
