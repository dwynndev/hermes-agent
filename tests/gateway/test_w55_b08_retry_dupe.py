"""B08 W54-F004: idempotent retry for the transcript retry queue (duplicate-retry).

The transcript retry queue re-appends a message whose prior write settled
ambiguously: ``append_message``'s commit raises SQLITE_IOERR after the row
actually landed (settlement unknown by design — ``hermes_state._execute_write``
deliberately propagates it), the gateway counts a failure and keeps the message
queued, and the next append (or a shutdown-spool replay) inserts the same
message dict again. Nothing in the messages DDL dedupes it, so the retry writes
a duplicate transcript row.

Fix contract (reviewer-confirmed minimal):
  * client-side uuid stamped at first enqueue, before any write attempt, so the
    key rides the message dict through the queue and the on-disk spool unchanged;
  * NULLABLE ``messages.client_msg_id`` column + UNIQUE partial index
    ``WHERE client_msg_id IS NOT NULL``;
  * the message insert becomes idempotent (INSERT OR IGNORE keyed on the id), so
    a retry after an ambiguous IOERR cannot duplicate.

Two guarantees under test:
  A. A commit-time failure AFTER the row landed, followed by a queue drain,
     leaves exactly ONE row (pre-fix: TWO) and session counters bumped once.
  B. A v30-era DB file (messages table created without ``client_msg_id``)
     upgrades cleanly — gaining the column AND the unique partial index —
     and the upgrade is idempotent (run twice).

Offline: SQLite on tmp_path only, deterministic, zero sleeps.
"""

import sqlite3

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_state import SessionDB

# The messages DDL as it stood at SCHEMA_VERSION 30, before client_msg_id
# existed. Hardcoded here (a schema-shape pin is the point of this test).
LEGACY_V30_MESSAGES_DDL = """CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT,
    display_identity BLOB,
    display_order INTEGER
);"""


@pytest.fixture
def store(tmp_path):
    return SessionStore(sessions_dir=tmp_path / "gw", config=GatewayConfig())


class _CommitBoomProxy:
    """Delegating connection proxy whose first commit() commits for real and
    THEN raises SQLITE_IOERR — settling the row with an unknown outcome, which
    is the exact ambiguity the retry queue must survive.

    The fired-once flag lives ON the proxy (``self._fired``), not on a dict the
    wrapper touches: the wrapper previously flipped ``armed["boom"]`` before
    delegating, which flipped the proxy's view of it too, so the poison never
    fired. The proxy owns its flag; it flips it only from inside commit()."""

    def __init__(self, conn, fired):
        self._conn = conn
        self._fired = False
        self._fired_publish = fired  # shared observer, flipped by commit() itself

    def commit(self):
        self._conn.commit()  # the row DOES land
        if not self._fired:
            self._fired = True
            self._fired_publish["fired"] = True
            raise sqlite3.OperationalError("disk I/O error")

    def __getattr__(self, name):
        return getattr(self._conn, name)


class TestRetryAfterAmbiguousCommitKeepsOneRow:
    def test_retry_after_commit_ioerr_does_not_duplicate(self, store, monkeypatch):
        """Commit lands the row and THEN raises SQLITE_IOERR; the queue retries the
        same message dict; exactly one row may exist afterwards."""
        db = store._db
        assert db is not None
        db.create_session("s1", "telegram", session_key="telegram:1")

        fired = {"fired": False}
        boom_proxy = None
        real_execute_write = db._execute_write

        def execute_write_with_one_commit_ioerr(fn, patience_s=None):
            nonlocal boom_proxy
            if boom_proxy is None:
                # Install the proxy exactly once, on the first write. Its own
                # fired-once flag governs the single poisoned commit.
                boom_proxy = _CommitBoomProxy(db._conn, fired)
                db._conn = boom_proxy
                try:
                    return real_execute_write(fn, patience_s=patience_s)
                finally:
                    db._conn = boom_proxy._conn
            return real_execute_write(fn, patience_s=patience_s)

        # sqlite3.Connection is immutable (instance AND class setattr both fail),
        # so the one-shot failure lives on the Python-level write seam instead.
        monkeypatch.setattr(db, "_execute_write", execute_write_with_one_commit_ioerr)

        first = {"role": "user", "content": "hello, survived the IOERR"}
        store.append_to_transcript("s1", first)  # row lands, commit raises: counts a failure, stays queued
        assert fired["fired"] is True, "the simulated commit failure never fired"

        # The next append drains the queue: retry the queued head, then the new message.
        store.append_to_transcript("s1", {"role": "assistant", "content": "the reply"})

        messages = db.get_messages("s1")
        assert sum(1 for m in messages if m.get("content") == "hello, survived the IOERR") == 1, (
            f"retry duplicated the ambiguously-settled row: {messages!r}"
        )
        assert len(messages) == 2, f"expected exactly the one-of-each pair: {messages!r}"
        # Counters must not double-count the ignored re-insert.
        assert db.get_session("s1")["message_count"] == 2


class TestLegacyV30Upgrade:
    def test_legacy_db_gains_column_and_index_idempotently(self, tmp_path):
        """A v30-era state.db (messages without client_msg_id) upgrades on open,
        and opening it a second time changes nothing (additive + idempotent)."""
        path = tmp_path / "state.db"
        conn = sqlite3.connect(path)
        try:
            conn.executescript(LEGACY_V30_MESSAGES_DDL)
            conn.commit()
        finally:
            conn.close()

        for _open in range(2):  # the upgrade must be a no-op the second time
            db = SessionDB(db_path=path)
            try:
                cols = {row[1] for row in db._conn.execute('PRAGMA table_info("messages")').fetchall()}
                assert "client_msg_id" in cols, (
                    f"legacy v30 DB did not gain client_msg_id on open #{_open + 1}: {sorted(cols)}"
                )
                index_row = db._conn.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'index' AND name = 'messages_client_msg_id'"
                ).fetchone()
                assert index_row is not None, (
                    f"unique partial index missing after open #{_open + 1}"
                )
                assert "UNIQUE" in index_row[0].upper()
                index_count = db._conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type = 'index' AND name = 'messages_client_msg_id'"
                ).fetchone()[0]
                assert index_count == 1, (
                    f"duplicate index creation after open #{_open + 1}: {index_count}"
                )
            finally:
                db.close()