"""Curator mutation-safety tests (T13-F3).

apply_automatic_transitions is the code that MOVES user skills between
active/stale/archived on a cron schedule. Its safety contracts — verified
against the source (agent/curator.py:305-380):

* pinned skills are NEVER transitioned, however old
* cron-referenced skills are NEVER transitioned (paused/disabled jobs count:
  the next fire must still find the skill)
* first-sight skills are seeded, not transitioned (clock starts at NOW)
* never-used skills get a grace floor: not archived while younger than
  stale_after_days; a stale-marked never-used young skill is reactivated
* aged-out skills archive; mid-aged active mark stale; recently-used stale
  reactivates

Driving the real function with a monkeypatched tools.skill_usage keeps this
freezegun-free (now is an injectable parameter).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import agent.curator as curator
from tools import skill_usage as su

NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)


def _row(name, *, state="active", last_activity=None, created=None,
         use_count=1, pinned=False, persisted=True):
    return {
        "name": name,
        "state": state,
        "last_activity_at": last_activity,
        "created_at": created,
        "use_count": use_count,
        "pinned": pinned,
        "_persisted": persisted,
    }


@pytest.fixture
def fake_skill_usage(monkeypatch):
    """Replace the skill_usage API the curator calls; record mutations."""
    calls = {"set_state": [], "archive": [], "seed": []}

    def set_state(name, state):
        calls["set_state"].append((name, state))

    def archive_skill(name):
        calls["archive"].append(name)
        return True, "ok"

    def seed_record_if_missing(name):
        calls["seed"].append(name)

    monkeypatch.setattr(su, "set_state", set_state)
    monkeypatch.setattr(su, "archive_skill", archive_skill)
    monkeypatch.setattr(su, "seed_record_if_missing", seed_record_if_missing)

    class _Fake:
        def __init__(self):
            self.calls = calls

        def install(self, rows):
            monkeypatch.setattr(su, "curated_report", lambda: list(rows))

    return _Fake()


@pytest.fixture
def fixed_windows(monkeypatch):
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 14)
    monkeypatch.setattr(curator, "get_archive_after_days", lambda: 60)


class TestNeverTouchGuards:
    def test_pinned_skill_never_archived(self, fake_skill_usage, fixed_windows):
        old = (NOW - timedelta(days=400)).isoformat()
        fake_skill_usage.install(
            [_row("pinned-old", last_activity=old, pinned=True)]
        )
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["archived"] == 0 and counts["marked_stale"] == 0
        assert fake_skill_usage.calls["archive"] == []

    def test_cron_referenced_skill_never_archived(self, monkeypatch,
                                                  fake_skill_usage, fixed_windows):
        monkeypatch.setattr(curator, "_cron_referenced_skills", lambda: {"cron-skill"})
        old = (NOW - timedelta(days=400)).isoformat()
        fake_skill_usage.install(
            [_row("cron-skill", last_activity=old)]
        )
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["archived"] == 0
        assert fake_skill_usage.calls["archive"] == []

    def test_first_sight_seeded_not_transitioned(self, fake_skill_usage,
                                                 fixed_windows):
        fake_skill_usage.install([_row("brand-new", persisted=False)])
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["seeded"] == 1
        assert counts["archived"] == 0 and counts["marked_stale"] == 0
        assert fake_skill_usage.calls["seed"] == ["brand-new"]


class TestGraceFloor:
    def test_never_used_young_skill_left_alone(self, fake_skill_usage,
                                               fixed_windows):
        recent = (NOW - timedelta(days=3)).isoformat()
        fake_skill_usage.install(
            [_row("fresh-never-used", created=recent, use_count=0,
                  last_activity=None)]
        )
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["archived"] == 0 and counts["marked_stale"] == 0

    def test_stale_marked_young_never_used_skill_is_reactivated(
        self, fake_skill_usage, fixed_windows
    ):
        recent = (NOW - timedelta(days=3)).isoformat()
        fake_skill_usage.install(
            [_row("mistakenly-stale", state="stale", created=recent,
                  use_count=0, last_activity=None)]
        )
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["reactivated"] == 1
        assert fake_skill_usage.calls["set_state"] == [
            ("mistakenly-stale", su.STATE_ACTIVE)
        ]


class TestAgingTransitions:
    def test_aged_out_active_skill_archives(self, fake_skill_usage,
                                            fixed_windows):
        old = (NOW - timedelta(days=90)).isoformat()
        fake_skill_usage.install([_row("old-skill", last_activity=old)])
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["archived"] == 1
        assert fake_skill_usage.calls["archive"] == ["old-skill"]

    def test_mid_age_active_skill_marks_stale(self, fake_skill_usage,
                                              fixed_windows):
        mid = (NOW - timedelta(days=30)).isoformat()  # > 14 stale, < 60 archive
        fake_skill_usage.install([_row("mid-skill", last_activity=mid)])
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["marked_stale"] == 1 and counts["archived"] == 0
        assert fake_skill_usage.calls["set_state"] == [
            ("mid-skill", su.STATE_STALE)
        ]

    def test_recently_used_stale_skill_reactivates(self, fake_skill_usage,
                                                   fixed_windows):
        recent = (NOW - timedelta(days=2)).isoformat()
        fake_skill_usage.install(
            [_row("revived", state="stale", last_activity=recent)]
        )
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["reactivated"] == 1
        assert fake_skill_usage.calls["set_state"] == [
            ("revived", su.STATE_ACTIVE)
        ]

    def test_created_at_anchors_never_active_skill(self, fake_skill_usage,
                                                   fixed_windows):
        # No last_activity and no created_at -> anchor falls back to NOW,
        # so the skill must NOT archive on a malformed/empty row.
        fake_skill_usage.install(
            [_row("orphan", last_activity=None, created=None)]
        )
        counts = curator.apply_automatic_transitions(now=NOW)
        assert counts["archived"] == 0
        assert fake_skill_usage.calls["archive"] == []
