"""Regression tests for W54 F006/F007/F008 — cron jobs.json races under the degraded lock.

W54-F006 (lost-update): under a DEGRADED `_jobs_lock()` section (flock timed out, another
process holds `.jobs.lock`), `_save_jobs_unlocked` blindly commits a payload derived from a
stale `load_jobs()` read — a concurrent writer's field update is lost, and a concurrently
deleted job is resurrected. The fix: stamp-CAS the section's `load_stamp` against the disk
stamp before commit and refuse loudly.

W54-F007 (read-check-write): `resume_job`/`trigger_job` derive `next_run_at` from an UNLOCKED
`resolve_job_ref` snapshot and later persist it — a concurrent schedule mutation between the
two sections is stamped over with a value derived from the OLD schedule. The fix: derive the
instant inside `update_job.apply` on the merged record under the lock.

W54-F008 (lost-update): `load_jobs()` auto-repair saves a legacy-shaped snapshot without
re-checking under the lock — a concurrent canonical write landing after the read is clobbered.
The fix: re-parse inside `_jobs_lock()` and repair only if the store still needs it.

Determinism strategy: no sleeps. F006 uses the held-flock second-fd pattern (an exclusive
flock from a second open() contends exactly like a foreign process, pushing `_jobs_lock` into
degraded mode) plus a wrapper that lands the foreign write into the gap the degraded lock
cannot close. F007/F008 inject the concurrent mutation at the exact unlock/lock boundary via
monkeypatch wrappers on `resolve_job_ref` / `_parse_jobs_file`.
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cron.jobs as jobs_mod

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


pytestmark = pytest.mark.skipif(fcntl is None, reason="flock semantics are POSIX-only")


def _hold_jobs_flock(path: Path, release: threading.Event, held: threading.Event):
    """Hold an exclusive flock on *path* from a separate fd until released.

    flock locks are per-open-file-description, so a second open() in the SAME
    process contends exactly like another process would.
    """
    fd = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        held.set()
        release.wait(timeout=30)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        fd.close()


@pytest.fixture()
def cron_store(tmp_path):
    """Isolated cron store per test (proven pattern: test_atomic_paused_creation)."""
    with jobs_mod.use_cron_store(tmp_path / "cron"):
        yield jobs_mod


@pytest.fixture()
def degraded_lock(monkeypatch):
    """Hold .jobs.lock from a second fd so _jobs_lock() degrades (in-process only).

    Yields (release_event, holder_thread); the test must release+join in a finally.
    """
    created = []

    def start():
        jobs_mod.ensure_dirs()
        lock_path = jobs_mod._jobs_lock_file()
        lock_path.touch()
        monkeypatch.setattr(jobs_mod, "_JOBS_LOCK_TIMEOUT_SECONDS", 1.0)
        release = threading.Event()
        held = threading.Event()
        holder = threading.Thread(
            target=_hold_jobs_flock, args=(lock_path, release, held), daemon=True
        )
        holder.start()
        assert held.wait(timeout=10), "test holder failed to take the flock"
        created.append((release, holder))
        return release, holder

    return start


def _write_disk_payload(jobs_list):
    """Overwrite jobs.json directly — the concurrent (unlocked) writer's commit."""
    jobs_file = jobs_mod._current_cron_store().jobs_file
    jobs_file.write_text(
        json.dumps({"jobs": jobs_list, "updated_at": jobs_mod._hermes_now().isoformat()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return jobs_file


# --------------------------------------------------------------------------- F006


def test_degraded_commit_refuses_stale_field_update(degraded_lock):
    """Concurrent field update + our stale save under a degraded lock must NOT lose the
    foreign change — the stale commit must fail loudly instead of blind-writing."""
    job = jobs_mod.create_job("field race", "every 1h", name="race-a")
    release, holder = degraded_lock()
    try:
        real_save = jobs_mod._save_jobs_unlocked

        def foreign_then_save(jobs, **kw):
            # A sibling process lands a field update in the gap the degraded lock can't close.
            disk = json.loads(jobs_mod._current_cron_store().jobs_file.read_text("utf-8"))
            for rec in disk["jobs"]:
                if rec["id"] == job["id"]:
                    rec["enabled"] = False
            _write_disk_payload(disk["jobs"])
            return real_save(jobs, **kw)

        original = jobs_mod._save_jobs_unlocked
        jobs_mod._save_jobs_unlocked = foreign_then_save
        try:
            with pytest.raises(RuntimeError, match="Refusing to commit cron jobs"):
                jobs_mod.update_job(job["id"], {"prompt": "stale writer"})
        finally:
            jobs_mod._save_jobs_unlocked = original

        on_disk = json.loads(
            jobs_mod._current_cron_store().jobs_file.read_text("utf-8"))["jobs"]
        rec = next(r for r in on_disk if r["id"] == job["id"])
        assert rec["enabled"] is False, "concurrent field update was clobbered"
        assert rec["prompt"] == "field race", "stale payload must not replace the record"
    finally:
        release.set()
        holder.join(timeout=10)


def test_degraded_commit_does_not_resurrect_deleted_job(degraded_lock):
    """A job deleted by a concurrent writer while we hold a STALE payload must stay
    deleted — the degraded stale commit must not resurrect it."""
    keep = jobs_mod.create_job("keep", "every 1h", name="keep")
    victim = jobs_mod.create_job("victim", "every 1h", name="victim")
    release, holder = degraded_lock()
    try:
        real_save = jobs_mod._save_jobs_unlocked

        def foreign_delete_then_save(jobs, **kw):
            disk = json.loads(jobs_mod._current_cron_store().jobs_file.read_text("utf-8"))
            disk["jobs"] = [rec for rec in disk["jobs"] if rec["id"] != victim["id"]]
            _write_disk_payload(disk["jobs"])
            return real_save(jobs, **kw)

        original = jobs_mod._save_jobs_unlocked
        jobs_mod._save_jobs_unlocked = foreign_delete_then_save
        try:
            with pytest.raises(RuntimeError, match="Refusing to commit cron jobs"):
                jobs_mod.update_job(keep["id"], {"prompt": "stale writer"})
        finally:
            jobs_mod._save_jobs_unlocked = original

        ids = {
            rec["id"]
            for rec in json.loads(
                jobs_mod._current_cron_store().jobs_file.read_text("utf-8"))["jobs"]
        }
        assert victim["id"] not in ids, "concurrently deleted job was resurrected"
        assert keep["id"] in ids, "concurrent delete must not nuke unrelated records"
    finally:
        release.set()
        holder.join(timeout=10)


# --------------------------------------------------------------------------- F007


def test_resume_derives_next_run_from_merged_record(monkeypatch):
    """A schedule mutation landing between resume's unlocked resolve and its locked write
    must win — next_run_at must derive from the MERGED record, not the stale snapshot."""
    now = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = jobs_mod.create_job("resume race", "*/10 * * * *", name="resume-race")
    jobs_mod.pause_job(job["id"], reason="race setup")

    real_resolve = jobs_mod.resolve_job_ref
    state = {"calls": 0}

    def resolving(ref):
        state["calls"] += 1
        snapshot = real_resolve(ref)
        if state["calls"] == 1 and snapshot is not None:
            # Concurrent schedule edit lands right after the unlocked snapshot.
            jobs_mod.update_job(snapshot["id"], {"schedule": "*/30 * * * *"})
        return snapshot

    monkeypatch.setattr(jobs_mod, "resolve_job_ref", resolving)

    resumed = jobs_mod.resume_job(job["id"])
    assert resumed is not None
    merged_schedule = jobs_mod.parse_schedule("*/30 * * * *")
    expected = jobs_mod.compute_next_run(merged_schedule)
    assert resumed["next_run_at"] == expected, (
        "next_run_at derived from the stale pre-mutation schedule"
    )
    # And the derivation must actually come from the record that WON the lock.
    assert resumed["schedule"]["expr"] == merged_schedule["expr"]
    assert jobs_mod.get_job(job["id"])["next_run_at"] == expected


def test_trigger_stamps_run_at_inside_the_lock(cron_store, monkeypatch):
    """trigger_job must mint manual_run_at/next_run_at inside the locked apply(), never on
    an unlocked pre-read — keeps the manual_run_at == next_run_at string-exact convention
    while binding the instant to the record version that is committed."""
    now = datetime(2026, 9, 12, 11, 0, 0, tzinfo=timezone.utc)
    calls = []

    def clock():
        calls.append(getattr(jobs_mod._jobs_lock_state, "depth", 0))
        return now

    monkeypatch.setattr("cron.jobs._hermes_now", clock)
    job = jobs_mod.create_job("trigger race", "every 1h", name="trigger-race")
    # create_job's own record-building clock read runs outside the lock by design — scope the
    # spy to the trigger path only (the mint inside apply + the locked save).
    calls.clear()

    triggered = jobs_mod.trigger_job(job["id"])
    assert triggered is not None
    assert calls, "expected at least one clock read"
    assert all(depth >= 1 for depth in calls), (
        "run-now instant was minted OUTSIDE the jobs lock (unlocked pre-read)"
    )
    assert triggered["manual_run_at"] == triggered["next_run_at"] == now.isoformat()


# --------------------------------------------------------------------------- F008


def test_repair_save_does_not_clobber_newer_state(cron_store, monkeypatch):
    """load_jobs' legacy auto-repair must NOT save a stale snapshot over a concurrent
    canonical write — the newer state survives on disk."""
    jobs_file = jobs_mod._current_cron_store().jobs_file
    legacy = {
        "jobs": {
            "repair-a": {
                "id": "repair-a", "name": "repair-a", "prompt": "legacy v1",
                "enabled": True, "state": "scheduled",
            },
        },
    }
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps(legacy), encoding="utf-8")

    newer_payload = {
        "jobs": [{
            "id": "repair-a", "name": "repair-a", "prompt": "newer v2",
            "enabled": True, "state": "scheduled",
        }],
    }
    real_parse = jobs_mod._parse_jobs_file
    state = {"calls": 0}

    def parsing(path):
        state["calls"] += 1
        if state["calls"] == 1:
            return legacy, False
        # A concurrent CANONICAL writer commits newer state right after our read.
        _write_disk_payload(newer_payload["jobs"])
        return newer_payload, False

    monkeypatch.setattr(jobs_mod, "_parse_jobs_file", parsing)

    jobs_mod.load_jobs()

    # Read the disk WITHOUT the wrapper to inspect what actually got committed.
    disk = real_parse(jobs_file)[0]["jobs"]
    assert any(rec.get("prompt") == "newer v2" for rec in disk), (
        "the concurrent canonical write was clobbered by the stale repair save"
    )
    assert all("repair-a" == rec.get("id") for rec in disk)
    assert state["calls"] >= 2, "repair decision must re-read under the lock"