"""B09 (W54-F005): racing writer vs ``unarchive_recoverable_session``.

LEAD ruling (authoritative, campaign-state.json b09_ruling): the contract CAS is
SOUND.  ``PATCH {archived: true}`` *without* ``end_reason`` is a state no-op
expressing no boundary intent — unarchive-after-it is a VALID linearization,
because the session's recoverable stamp (``ws_orphan_reap``) is still on the
row, so resurrecting it is correct.  A FULL deliberate archive writes a
BOUNDARY ``end_reason`` (``db.end_session`` — gateway/platforms/api_server.py
PATCH handler).  The real pre-fix damage: recovery's second write
``UPDATE sessions SET ended_at = NULL, end_reason = NULL`` WIPES a boundary
stamp committed inside the check->write window, silently converting a
deliberate archive into an accidental-looking one.

Pre-fix ``unarchive_recoverable_session`` is check-then-two-writes
(hermes_state_sessions.py): (1) read tip, verify ``end_reason`` recoverable;
(2) ``set_session_archived(False)``; (3) blast ``ended_at/end_reason`` to
NULL on the tip.  A writer landing between (1) and (3) loses its boundary.

The race is forced deterministically with ZERO sleeps: a wrapper on the
``_execute_write`` seam (present in both pre- and post-fix recovery — the
pre-fix first write is ``set_session_archived``'s, the post-fix one is the
atomic CAS transaction) gate-fires on the recovery thread's first write,
holds it on an ``Event`` while the racing writer completes the FULL deliberate
archive, then lets the recovery proceed.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_reaped_row(db: SessionDB, session_id: str) -> None:
    """The ruled pre-image: session archived by the ws-orphan reaper with a
    recoverable end stamp and an open ended_at (the boundary write must be
    able to land, per the LEAD's ruled model)."""
    db._write_sql(
        "INSERT INTO sessions (id, source, started_at, archived, end_reason, ended_at) "
        "VALUES (?, 'desktop', ?, 1, ?, NULL)",
        (session_id, time.time(), "ws_orphan_reap"),
    )


class _RaceGate:
    """Holds the recovery thread at its first ``_execute_write`` (i.e. after
    the recoverability check) until the racing writer finishes."""

    def __init__(self, db: SessionDB) -> None:
        self.main_thread = threading.current_thread()
        self.real_execute_write = db._execute_write
        self.entered = threading.Event()
        self.writer_done = threading.Event()
        self.writer_error: list = []
        self._fired = False

    def install(self, db: SessionDB) -> None:
        db._execute_write = self._gated  # type: ignore[assignment]

    def _gated(self, fn, patience_s=None):
        if threading.current_thread() is self.main_thread and not self._fired:
            self._fired = True
            self.entered.set()
            if not self.writer_done.wait(10):
                raise AssertionError("racing writer never finished before recovery resumed")
        return self.real_execute_write(fn, patience_s)


def _racing_writer(db: SessionDB, gate: _RaceGate, session_id: str, *, with_boundary: bool) -> None:
    try:
        if not gate.entered.wait(10):
            raise AssertionError("recovery never reached its first write")
        # PATCH {archived: true}: the desktop flag flip (state no-op here — row is archived).
        db.set_session_archived(session_id, True)
        if with_boundary:
            # PATCH body end_reason -> db.end_session: the deliberate BOUNDARY stamp.
            db.end_session(session_id, "session_reset")
    except BaseException as exc:  # noqa: BLE001 — surfaced to the test, not lost in a thread
        gate.writer_error.append(exc)
    finally:
        gate.writer_done.set()


class TestBoundaryArchiveLandsInsideRecoveryWindow:
    """FULL deliberate archive (archived + end_reason boundary) racing the
    unarchive-recovery: the boundary stamp must SURVIVE and recovery must
    abort.  Pre-fix this fails: recovery returns True and the boundary stamp
    (plus ended_at) is wiped to NULL by the second write."""

    def test_boundary_stamp_survives_recovery_race(self, db):
        session_id = "race-sess-boundary"
        _seed_reaped_row(db, session_id)
        gate = _RaceGate(db)
        gate.install(db)
        writer = threading.Thread(
            target=_racing_writer, args=(db, gate, session_id), kwargs={"with_boundary": True},
        )
        writer.start()
        try:
            recovered = db.unarchive_recoverable_session(session_id)
        finally:
            writer.join(10)
        assert not writer.is_alive(), "racing writer hung"
        assert not gate.writer_error, gate.writer_error

        row = db.get_session(session_id)
        # Post-fix (contract): the boundary stamp landed inside the window →
        # the recoverable-row CAS matches nothing → recovery returns False and
        # the deliberate archive stays intact.
        assert recovered is False, "recovery must NOT resurrect a deliberate archive"
        assert row["archived"] == 1, "deliberate archive must stay archived"
        assert row["end_reason"] == "session_reset", "boundary stamp was wiped (W54-F005)"
        assert row["ended_at"] is not None, "boundary ended_at was wiped (W54-F005)"


class TestArchiveFlagWithoutBoundaryIsValidLinearization:
    """The ruling's GREEN pin, both ways: a concurrent PATCH-style
    ``set_session_archived(True)`` WITHOUT ``end_reason`` carries no boundary
    intent — the recoverable stamp is still on the row, so recovery may
    legitimately unarchive it.  Asserts no crash + a consistent final state
    (unarchived, stamp cleared).  Must pass pre AND post fix."""

    def test_archiveless_patch_race_is_legitimately_unarchived(self, db):
        session_id = "race-sess-patch"
        _seed_reaped_row(db, session_id)
        gate = _RaceGate(db)
        gate.install(db)
        writer = threading.Thread(
            target=_racing_writer, args=(db, gate, session_id), kwargs={"with_boundary": False},
        )
        writer.start()
        try:
            recovered = db.unarchive_recoverable_session(session_id)
        finally:
            writer.join(10)
        assert not writer.is_alive(), "racing writer hung"
        assert not gate.writer_error, gate.writer_error

        row = db.get_session(session_id)
        assert recovered is True, "a no-boundary PATCH race is a valid linearization for recovery"
        assert row["archived"] == 0, "recovery must land the unarchive"
        assert row["end_reason"] is None, "accidental stamp should be cleared on recovery"
        assert row["ended_at"] is None, "accidental end stamp should be cleared on recovery"