"""Tests for the worker context's restartable single-task helper."""

from __future__ import annotations

import anyio
import anyio.lowlevel
import pytest

from anycorn.task_group import TaskGroup
from anycorn.worker_context import AnyioSingleTask, WorkerContext


@pytest.mark.anyio
async def test_single_task_reschedules_itself_repeatedly() -> None:
    """A task that restarts its own SingleTask must keep running, not stop after one run.

    This is the shape the QUIC connection timer has: _handle_timer waits for the
    deadline, does its work, then send_all calls restart on the very SingleTask
    driving it. If restart cancels the running handle before the replacement is
    armed, that self-cancellation tears the reschedule down at restart's own await
    and the timer fires exactly once - which on a lossy link stops QUIC
    retransmitting after the first attempt. It reproduces here without any sockets:
    the giveaway is a real await (the timer's own sleep) before the reschedule, so
    the replacement has a checkpoint at which the stale cancellation can hit it.
    """
    fires: list[int] = []
    target = 5
    done = anyio.Event()

    async with TaskGroup() as task_group:
        single_task = AnyioSingleTask()

        async def action() -> None:
            await anyio.sleep(0.01)  # the timer waiting out its deadline
            fires.append(len(fires))
            await anyio.lowlevel.checkpoint()  # stands in for send_all's awaited sends
            if len(fires) < target:
                await single_task.restart(task_group, action)
            else:
                done.set()

        await single_task.restart(task_group, action)

        with anyio.fail_after(5):
            await done.wait()

        await single_task.stop()

    assert len(fires) == target


@pytest.mark.anyio
async def test_mark_request_signals_termination_once_the_limit_is_passed() -> None:
    """Once more requests have been served than max_requests, the worker is told to stop."""
    context = WorkerContext(max_requests=2)
    await context.mark_request()  # 1
    await context.mark_request()  # 2 - still at the limit, not past it
    assert not context.terminate.is_set()
    await context.mark_request()  # 3 - now past the limit
    assert context.terminate.is_set()


@pytest.mark.anyio
async def test_mark_request_without_a_limit_never_terminates() -> None:
    """With no max_requests the counter is not even tracked and termination never fires."""
    context = WorkerContext(max_requests=None)
    for _ in range(10):
        await context.mark_request()
    assert not context.terminate.is_set()
    assert context.requests == 0  # the counter is left untouched when there is no limit
