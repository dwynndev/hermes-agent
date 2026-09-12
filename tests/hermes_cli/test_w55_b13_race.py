"""W55-B13 (W54-F020): the wedged-gateway escalation SIGTERM must carry the start-time fingerprint.

Between the liveness witness and the signal, a wedged gateway can die on its own and the OS
recycles its PID to an unrelated process. ``_escalate_wedged_gateway`` captures the gateway's
start-time fingerprint at entry (gateway.py:508) but its first (non-force) SIGTERM does not pass
it to ``terminate_pid`` — on a recycled PID the innocent process is signaled. The fix passes the
already-captured fingerprint (one kwarg); ``terminate_pid``'s guard refuses on mismatch.

Simulation (deterministic, zero sleeps): the capture-time probe reports the wedged gateway's
start time, the kill-time probe reports a DIFFERENT start time (an unrelated process reused the
pid), and the pid is gone by the time the escalation's wait phase runs. A termination signal
attempted at the pid in any path is a failed test.
"""
import os
import signal

import pytest

from hermes_cli import gateway as gateway_module
from gateway import status as status_module

WEDGED_PID = 999999
CAPTURED_START = 16_000_000  # fingerprint of the wedged gateway as last witnessed (capture, :508)
RECYCLED_START = 17_000_000  # fingerprint of the unrelated process that reused the pid


@pytest.fixture()
def recycled_pid(monkeypatch):
    """Deterministic PID-recycle simulation: capture-time identity != kill-time identity."""
    kills: list[tuple[int, int]] = []

    real_kill = os.kill

    def fake_kill(pid, sig):
        if pid == WEDGED_PID and sig in (
            signal.SIGTERM,
            getattr(signal, "SIGKILL", signal.SIGTERM),
        ):
            kills.append((pid, sig))
            return None
        return real_kill(pid, sig)

    monkeypatch.setattr(status_module.os, "kill", fake_kill)

    real_get = status_module.get_process_start_time
    real_get_private = status_module._get_process_start_time
    monkeypatch.setattr(
        status_module,
        "get_process_start_time",
        lambda pid: CAPTURED_START if pid == WEDGED_PID else real_get(pid),
    )
    monkeypatch.setattr(
        status_module,
        "_get_process_start_time",
        lambda pid: RECYCLED_START if pid == WEDGED_PID else real_get_private(pid),
    )

    real_pid_exists = status_module._pid_exists
    monkeypatch.setattr(
        status_module,
        "_pid_exists",
        lambda pid: False if pid == WEDGED_PID else real_pid_exists(pid),
    )

    return kills


def test_escalate_wedged_gateway_never_signals_a_recycled_pid(recycled_pid):
    kills = recycled_pid
    result = gateway_module._escalate_wedged_gateway(
        WEDGED_PID, term_grace=0.0, kill_wait=0.0
    )
    assert kills == [], (
        f"termination signal attempted at recycled PID {WEDGED_PID}: {kills}"
    )
    assert result is True