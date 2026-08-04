"""Schedule normalization/dedup tests for cron.jobs (T13-F4).

parse_schedule is the single entry point for every user-supplied schedule
string (duration one-shot, 'every X' interval, 5-field cron, ISO timestamp).
A misparse means a job that never fires, fires at the wrong wall-clock time,
or silently becomes the wrong KIND of job. Pinned contracts:

* kind classification: 'every 30m' -> interval, '0 9 * * *' -> cron,
  '30m' -> once, '2026-08-04T14:00' -> once
* interval minutes are exact integers and round-trip to display
* invalid cron expressions raise ValueError (never stored)
* naive ISO timestamps anchor to the CONFIGURED Hermes timezone (#51021) —
  not server-local — so wall-clock intent survives timezone mismatch
* 'Z' timestamps keep their explicit UTC offset
* parse_duration unit multipliers and rejection of malformed input
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import cron.jobs as jobs
from cron.jobs import parse_duration, parse_schedule

FIXED_NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED_NOW)


class TestParseDuration:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("30m", 30),
            ("2h", 120),
            ("1d", 1440),
            ("90 minutes", 90),
            ("3 hrs", 180),
            (" 5m ", 5),
        ],
    )
    def test_unit_multipliers(self, inp, expected):
        assert parse_duration(inp) == expected

    @pytest.mark.parametrize("bad", ["", "abc", "30", "m30", "-5m", "5w"])
    def test_malformed_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_duration(bad)


class TestIntervalSchedules:
    def test_every_pattern_is_interval(self, fixed_clock):
        r = parse_schedule("every 30m")
        assert r["kind"] == "interval"
        assert r["minutes"] == 30
        assert r["display"] == "every 30m"

    def test_every_hours(self, fixed_clock):
        r = parse_schedule("every 2h")
        assert r["kind"] == "interval" and r["minutes"] == 120


class TestCronSchedules:
    def test_five_field_cron_expression(self, fixed_clock):
        r = parse_schedule("0 9 * * *")
        assert r["kind"] == "cron"
        assert r["expr"] == "0 9 * * *"

    def test_invalid_cron_expression_rejected(self, fixed_clock):
        with pytest.raises(ValueError, match="[Ii]nvalid cron"):
            parse_schedule("99 99 * * *")

    def test_named_weekday_not_accepted_as_cron(self, fixed_clock):
        # Current contract: only numeric/star/comma/dash/slash fields count as
        # cron; named entries fall through and error out rather than silently
        # mis-scheduling.
        with pytest.raises(ValueError):
            parse_schedule("0 9 * * mon")


class TestOneShotSchedules:
    def test_duration_is_once_from_now(self, fixed_clock):
        r = parse_schedule("30m")
        assert r["kind"] == "once"
        run_at = datetime.fromisoformat(r["run_at"])
        assert run_at == FIXED_NOW + timedelta(minutes=30)
        assert run_at.tzinfo is not None

    def test_naive_iso_timestamp_anchored_to_configured_tz(self, fixed_clock):
        # #51021: naive 'T14:00' must be interpreted in the Hermes-configured
        # timezone (fixed_clock uses UTC here), NOT server-local.
        r = parse_schedule("2026-08-04T14:00")
        assert r["kind"] == "once"
        run_at = datetime.fromisoformat(r["run_at"])
        assert run_at.tzinfo is not None
        assert run_at == datetime(2026, 8, 4, 14, 0, 0, tzinfo=timezone.utc)

    def test_explicit_utc_z_keeps_offset(self, fixed_clock):
        r = parse_schedule("2026-08-04T14:00:00Z")
        assert r["kind"] == "once"
        run_at = datetime.fromisoformat(r["run_at"])
        assert run_at == datetime(2026, 8, 4, 14, 0, 0, tzinfo=timezone.utc)

    def test_invalid_timestamp_rejected(self, fixed_clock):
        with pytest.raises(ValueError, match="[Ii]nvalid timestamp"):
            parse_schedule("2026-99-99T99:00")


class TestGarbageInput:
    def test_unparseable_schedule_raises_with_usage(self, fixed_clock):
        with pytest.raises(ValueError, match="Invalid schedule"):
            parse_schedule("sometime next week")

    def test_whitespace_only_rejected(self, fixed_clock):
        with pytest.raises(ValueError):
            parse_schedule("   ")
