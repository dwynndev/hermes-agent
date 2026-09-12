"""W54-F010 (B11) — auxiliary transcript writers must honor the active-turn lease fence.

Before the fix, shutdown-flush spool recovery and the TUI model-switch marker append wrote
transcript rows into a session whose cross-process turn lease is held by a live turn, letting
auxiliary rows interleave with an in-flight transcript. After the fix both call sites pass
``reject_active_turn_lease=True``, so a held lease rejects the write: the spool file stays on
disk (retried later) and no message row is added. Deterministic — the lease row is created
through the public lease machinery; no sleeps or real turns involved.
"""
import json
import logging
import os
from types import SimpleNamespace

import pytest

from gateway import shutdown_flush
from hermes_state import SessionDB
from tui_gateway.server import _append_model_switch_marker, _is_model_switch_marker

SESSION_ID = "b11-race-sess"
HOLDER = f"pid={os.getpid()}:turn=live"
SPOOLED_TEXT = "spooled-while-lease-held"


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    handle.create_session(SESSION_ID, source="test")
    return handle


def _hold_turn_lease(db):
    assert db.try_acquire_session_turn_lease(SESSION_ID, HOLDER, ttl_seconds=120)


def _contents(db, needle):
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", (SESSION_ID,)
        ).fetchall()
    return [r["content"] for r in rows if needle in str(r["content"] or "")]


class TestShutdownFlushHonorsTurnLease:
    def test_spool_preserved_and_row_rejected_under_active_turn_lease(
        self, db, tmp_path, monkeypatch
    ):
        flush_dir = tmp_path / "pending_messages"
        flush_dir.mkdir()
        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: flush_dir)
        spool_path = flush_dir / "pending-deadbeef.json"
        spool_path.write_text(
            json.dumps({
                "session_key": SESSION_ID,
                "reason": shutdown_flush.TRANSCRIPT_CAP_DROP_REASON,
                "data": {"session_id": SESSION_ID,
                         "message": {"role": "user", "content": SPOOLED_TEXT}},
            })
        )
        _hold_turn_lease(db)

        recovered = shutdown_flush.recover_pending_to_db(session_db=db)

        assert recovered == 0
        assert spool_path.exists()
        assert _contents(db, SPOOLED_TEXT) == []


class TestTuiModelSwitchMarkerHonorsTurnLease:
    def test_marker_row_rejected_under_active_turn_lease(self, db, caplog):
        session = {"session_key": SESSION_ID, "agent": SimpleNamespace(_session_db=db)}
        _hold_turn_lease(db)

        with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
            _append_model_switch_marker(session, model="gpt-test", provider="openai")

        assert _contents(db, "active model for this chat has changed") == []
        # The in-memory marker survives (self-healing retry on the next marker switch).
        assert any(_is_model_switch_marker(entry) for entry in session["history"])