"""B08 fix-round regression tests (R2): client_msg_id is enqueue-scoped metadata.

Round-1 review BLOCKING findings these lock in (see B08-review.json round 1):

  B1 clone paths — ``_clone_message_rows`` copied ``client_msg_id`` into the
     re-sequenced clones while the soft-archived originals still held the id;
     the GLOBAL ``messages_client_msg_id`` UNIQUE partial index rejected the
     clone INSERT (raw SQL, no ON CONFLICT clause) with IntegrityError, so
     ``archive_and_compact`` with a watermark raised instead of re-sequencing.

  B2 replace/rewind re-inserts — ``replace_messages`` in soft-archive mode
     (``archive_dropped=True``, the mode rewind/edit/truncate use) re-inserts
     dicts from ``get_messages``, whose SELECT * now exposes ``client_msg_id``;
     the re-insert hit ON CONFLICT DO NOTHING whose key still lived on the
     soft-archived originals, silently swallowing the live transcript rows.

  M1 counter — ``_insert_message_rows`` counted ``inserted += 1``
     unconditionally, so ``append_messages_batch`` reported rows it never
     landed when a duplicate key was swallowed.

Contract under test: row copies (clones, re-inserts fed get_messages output)
must NOT carry ``client_msg_id`` — they insert NULL, which the partial index
exempts — while the enqueue path keeps the key for retry idempotency, and the
batch counter reports the rows that actually landed.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-clone", source="cli")
    yield d
    d.close()


class TestWatermarkCompactionClonesStampedRows:
    def test_archive_and_compact_watermark_no_integrity_error_clones_live(self, db):
        """Clones of stamped tail rows must insert with a NULL client_msg_id.

        Pre-fix this raised ``sqlite3.IntegrityError: UNIQUE constraint failed:
        messages.client_msg_id`` inside archive_and_compact (clone INSERT vs
        the soft-archived originals still holding the key).
        """
        db.append_message("sess-clone", "user", "pre-compaction", client_msg_id="t:a")
        watermark = db.get_active_message_watermark("sess-clone")
        assert watermark == 1
        db.append_message("sess-clone", "user", "tail-1", client_msg_id="t:b")
        db.append_message("sess-clone", "user", "tail-2", client_msg_id="t:c")

        db.archive_and_compact(
            "sess-clone",
            [{"role": "assistant", "content": "compacted summary"}],
            watermark=watermark,
        )

        live = db.get_messages("sess-clone")
        contents = [m["content"] for m in live]
        assert len(live) == 3, f"expected summary + 2 live clones: {live!r}"
        assert contents.count("compacted summary") == 1
        assert contents.count("tail-1") == 1 and contents.count("tail-2") == 1
        # Counters match the live set exactly (clones counted, originals not).
        assert db.get_session("sess-clone")["message_count"] == 3
        # Originals are archived, not deleted-and-replaced: 3 archived + 3 live.
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 6


class TestReplaceMessagesRoundtripPreservesRows:
    def test_replace_messages_soft_archive_mode_roundtrip_keeps_live_rows(self, db):
        """get_messages dicts re-inserted via replace_messages must all land.

        Pre-fix: get_messages exposes client_msg_id, the soft-archived
        originals keep it, and the re-insert ON CONFLICT DO NOTHING swallowed
        every row — live transcript went empty while the counters still
        claimed 2 (silent transcript loss on rewind/edit/truncate paths).
        """
        db.append_message("sess-clone", "user", "first", client_msg_id="t:1")
        db.append_message("sess-clone", "assistant", "second", client_msg_id="t:2")

        msgs = db.get_messages("sess-clone")
        assert [m.get("client_msg_id") for m in msgs] == ["t:1", "t:2"], (
            "get_messages must expose client_msg_id (the round-trip carries it)"
        )

        db.replace_messages("sess-clone", msgs, archive_dropped=True)

        live = db.get_messages("sess-clone")
        assert len(live) == 2, f"soft-archive replace swallowed live rows: {live!r}"
        assert [m["content"] for m in live] == ["first", "second"]
        # Counters agree with the live set.
        assert db.get_session("sess-clone")["message_count"] == 2


class TestAppendMessagesBatchCountsLandedRows:
    def test_append_messages_batch_returns_rows_actually_landed(self, db):
        """A swallowed duplicate must not inflate the reported inserted count.

        Pre-fix: _insert_message_rows counted inserted += 1 unconditionally,
        so a batch whose head duplicated an already-settled client_msg_id
        reported 2 while only 1 row landed (and bumped counters by 2).
        """
        db.append_message("sess-clone", "user", "already settled", client_msg_id="t:x")

        landed = db.append_messages_batch("sess-clone", [
            {"role": "user", "content": "already settled", "client_msg_id": "t:x"},
            {"role": "assistant", "content": "fresh"},
        ])

        assert landed == 1, f"reported {landed} rows, 1 landed"
        assert db.get_session("sess-clone")["message_count"] == 2