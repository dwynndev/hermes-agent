"""Next-run computation + one-shot lifecycle tests for cron.jobs (T13-F11).

compute_next_run decides WHEN every job fires next — a wrong computation means
a job that never fires, double-fires, or fast-forwards over a missed run.
Pinned contracts (verified against cron/jobs.py:708-831):

* one-shot: eligible within the grace window, NEVER again after last_run_at,
  past-grace one-shots return None (not resurrected)
* interval: first run is now+minutes; later runs anchor to LAST RUN + minutes
  (crash-restart must not drift the phase); malformed last_run_at falls back
  to now+minutes instead of crashing the ticker
* cron: anchored to last_run_at when present (restart-safe), malformed
  last_run_at falls back to now
* _compute_grace_seconds: half-period clamped to [120s, 7200s] so daily jobs
  catch up after <=2h downtime while frequent jobs fast-forward
* degenerate schedules (non-dict, no kind, missing minutes/expr) return None
  — the ticker must skip, not crash
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import cron.jobs as jobs
from cron.jobs import _compute_grace_seconds, _recoverable_oneshot_run_at, compute_next_run

FIXED_NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)


def _parse(result: str | None) -> datetime:
    """Assert the computation produced a timestamp, then parse it."""
    assert result is not None, "expected a next-run timestamp, got None"
    return datetime.fromisoformat(result)


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED_NOW)


class TestOneShotLifecycle:
    def test_future_oneshot_is_eligible(self, fixed_clock):
        run_at = (FIXED_NOW + timedelta(minutes=10)).isoformat()
        assert compute_next_run({"kind": "once", "run_at": run_at}) == run_at

    def test_just_past_oneshot_within_grace_still_fires(self, fixed_clock):
        # Created a few seconds after the requested minute — next tick catches it.
        grace = jobs.ONESHOT_GRACE_SECONDS
        run_at = (FIXED_NOW - timedelta(seconds=grace - 1)).isoformat()
        assert compute_next_run({"kind": "once", "run_at": run_at}) == run_at

    def test_past_grace_oneshot_is_dead(self, fixed_clock):
        grace = jobs.ONESHOT_GRACE_SECONDS
        run_at = (FIXED_NOW - timedelta(seconds=grace + 60)).isoformat()
        assert compute_next_run({"kind": "once", "run_at": run_at}) is None

    def test_oneshot_never_fires_twice(self, fixed_clock):
        run_at = (FIXED_NOW + timedelta(minutes=10)).isoformat()
        sched = {"kind": "once", "run_at": run_at}
        assert compute_next_run(sched, last_run_at=FIXED_NOW.isoformat()) is None

    def test_oneshot_missing_run_at_is_dead(self, fixed_clock):
        assert compute_next_run({"kind": "once"}) is None

    def test_oneshot_garbage_run_at_is_dead(self, fixed_clock):
        assert _recoverable_oneshot_run_at(
            {"kind": "once", "run_at": "not-a-date"}, FIXED_NOW
        ) is None


class TestIntervalNextRun:
    def test_first_run_is_now_plus_interval(self, fixed_clock):
        r = compute_next_run({"kind": "interval", "minutes": 30})
        assert _parse(r) == FIXED_NOW + timedelta(minutes=30)

    def test_subsequent_run_anchors_to_last_run(self, fixed_clock):
        # Crash-restart phase preservation: next = last + minutes, NOT now +
        # minutes (which would drift every restart).
        last = (FIXED_NOW - timedelta(minutes=12)).isoformat()
        r = compute_next_run({"kind": "interval", "minutes": 30}, last_run_at=last)
        assert _parse(r) == datetime.fromisoformat(last) + timedelta(minutes=30)

    def test_malformed_last_run_falls_back_to_now(self, fixed_clock):
        r = compute_next_run(
            {"kind": "interval", "minutes": 15}, last_run_at="garbage"
        )
        assert _parse(r) == FIXED_NOW + timedelta(minutes=15)

    def test_missing_minutes_is_dead(self, fixed_clock):
        assert compute_next_run({"kind": "interval"}) is None


class TestCronNextRun:
    def test_no_last_run_uses_now_base(self, fixed_clock):
        r = compute_next_run({"kind": "cron", "expr": "0 13 * * *"})
        assert _parse(r) == datetime(2026, 8, 4, 13, 0, tzinfo=timezone.utc)

    def test_last_run_anchor_beats_now(self, fixed_clock):
        # Daily job that ran at 13:00 yesterday: next must be 13:00 today,
        # anchored to last_run_at, not the restart instant.
        last = datetime(2026, 8, 3, 13, 0, tzinfo=timezone.utc).isoformat()
        r = compute_next_run({"kind": "cron", "expr": "0 13 * * *"}, last_run_at=last)
        assert _parse(r) == datetime(2026, 8, 4, 13, 0, tzinfo=timezone.utc)

    def test_missing_expr_is_dead(self, fixed_clock):
        assert compute_next_run({"kind": "cron"}) is None


class TestGraceComputation:
    def test_interval_half_period(self):
        # 30m job -> 900s grace
        assert _compute_grace_seconds({"kind": "interval", "minutes": 30}) == 900

    def test_interval_floor_clamp(self):
        # 1m job -> 30s unclamped, floor is 120s
        assert _compute_grace_seconds({"kind": "interval", "minutes": 1}) == 120

    def test_interval_ceiling_clamp(self):
        # 1d job -> 12h unclamped, ceiling is 7200s (2h catch-up window)
        assert _compute_grace_seconds({"kind": "interval", "minutes": 1440}) == 7200

    def test_unknown_kind_gets_minimum(self):
        assert _compute_grace_seconds({"kind": "once"}) == 120
        assert _compute_grace_seconds({}) == 120


class TestDegenerateSchedules:
    @pytest.mark.parametrize("bad", [None, "30m", [], {"minutes": 30}, {}])
    def test_non_or_malformed_schedule_returns_none(self, bad, fixed_clock):
        assert compute_next_run(bad) is None
