"""Tests for the structured JSONL error ledger (T6).

Covers: agent/error_ledger.py (formatter, read_recent, summarize),
the hermes_logging root-handler choke point, and the cmd_errors CLI.
"""

from __future__ import annotations

import json
import logging
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.error_ledger import (
    LEDGER_FILENAME,
    ErrorLedgerFormatter,
    ledger_path,
    read_recent,
    summarize,
)


def _make_record(
    msg: str,
    *,
    level: int = logging.ERROR,
    name: str = "test.logger",
    exc_info=None,
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


class TestErrorLedgerFormatter:
    def test_basic_json_line(self):
        line = ErrorLedgerFormatter().format(_make_record("boom"))
        entry = json.loads(line)
        assert entry["level"] == "ERROR"
        assert entry["logger"] == "test.logger"
        assert entry["message"] == "boom"
        assert "ts" in entry

    def test_category_and_detail_from_extra(self):
        rec = _make_record(
            "API failed",
            extra={"error_category": "api", "error_detail": "provider=x"},
        )
        entry = json.loads(ErrorLedgerFormatter().format(rec))
        assert entry["category"] == "api"
        assert entry["detail"] == "provider=x"

    def test_exception_info_captured(self):
        try:
            raise ValueError("kapow")
        except ValueError:
            import sys

            rec = _make_record("job failed", exc_info=sys.exc_info())
        entry = json.loads(ErrorLedgerFormatter().format(rec))
        assert entry["exc_type"] == "ValueError"
        assert "kapow" in entry["traceback"]

    def test_session_tag_extracted(self):
        rec = _make_record("x")
        rec.session_tag = " [sess123]"
        entry = json.loads(ErrorLedgerFormatter().format(rec))
        assert entry["session"] == "sess123"

    def test_secret_redaction(self):
        rec = _make_record("leaked key sk-1234567890abcdef1234567890abcdef")
        line = ErrorLedgerFormatter().format(rec)
        # redact_sensitive_text must have scrubbed the raw secret
        assert "sk-1234567890abcdef1234567890abcdef" not in line

    def test_oversized_record_clipped_and_flagged(self):
        rec = _make_record("x" * 10_000)
        line = ErrorLedgerFormatter().format(rec)
        entry = json.loads(line)
        assert entry.get("truncated") is True

    def test_clipped_record_always_fits_atomic_bound(self):
        # Regression: first-stage clipping alone didn't re-check the bound —
        # a huge extra=detail field kept the line above _MAX_RECORD_BYTES,
        # breaking the documented O_APPEND atomicity invariant.
        rec = _make_record("y" * 10_000)
        rec.error_detail = "z" * 10_000
        try:
            raise RuntimeError("w" * 5_000)
        except RuntimeError:
            import sys

            rec.exc_info = sys.exc_info()
        line = ErrorLedgerFormatter().format(rec)
        assert len(line.encode("utf-8")) <= 4000
        entry = json.loads(line)
        assert entry.get("truncated") is True
        assert "detail" not in entry  # dropped in the second reduction stage
        assert len(line.encode("utf-8")) <= 4096  # PIPE_BUF-safe bound


class TestReadRecent:
    @pytest.fixture()
    def ledger_dir(self, tmp_path: Path) -> Path:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        return log_dir

    def _write(self, log_dir: Path, entries: list[dict]) -> None:
        path = log_dir / LEDGER_FILENAME
        with open(path, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

    def test_missing_ledger_returns_empty(self, ledger_dir: Path):
        assert read_recent(limit=5, log_dir=ledger_dir) == []

    def test_newest_first_and_limit(self, ledger_dir: Path):
        now = datetime.now(tz=timezone.utc)
        # Ledger appends chronologically: oldest first in the file.
        entries = [
            {"ts": (now - timedelta(minutes=5 - i)).isoformat(), "message": f"m{i}"}
            for i in range(5)  # m0 oldest ... m4 newest
        ]
        self._write(ledger_dir, entries)
        recs = read_recent(limit=3, log_dir=ledger_dir)
        assert [r["message"] for r in recs] == ["m4", "m3", "m2"]

    def test_category_filter(self, ledger_dir: Path):
        now = datetime.now(tz=timezone.utc).isoformat()
        self._write(ledger_dir, [
            {"ts": now, "message": "a", "category": "api"},
            {"ts": now, "message": "c", "category": "cron"},
            {"ts": now, "message": "g"},
        ])
        recs = read_recent(limit=10, category="cron", log_dir=ledger_dir)
        assert [r["message"] for r in recs] == ["c"]

    def test_since_window_excludes_old(self, ledger_dir: Path):
        now = datetime.now(tz=timezone.utc)
        self._write(ledger_dir, [
            {"ts": now.isoformat(), "message": "fresh"},
            {"ts": (now - timedelta(hours=5)).isoformat(), "message": "stale"},
        ])
        recs = read_recent(limit=10, since_seconds=3600, log_dir=ledger_dir)
        assert [r["message"] for r in recs] == ["fresh"]

    def test_torn_lines_skipped(self, ledger_dir: Path):
        now = datetime.now(tz=timezone.utc).isoformat()
        path = ledger_dir / LEDGER_FILENAME
        path.write_text(
            json.dumps({"ts": now, "message": "good"}) + "\n"
            + '{"ts": "broken json\n'
            + "\n"
            + json.dumps({"ts": now, "message": "also good"}) + "\n",
            encoding="utf-8",
        )
        recs = read_recent(limit=10, log_dir=ledger_dir)
        assert sorted(r["message"] for r in recs) == ["also good", "good"]


class TestSummarize:
    def test_missing_ledger(self, tmp_path: Path):
        stats = summarize(log_dir=tmp_path / "logs")
        assert stats["ledger_exists"] is False
        assert stats["total"] == 0

    def test_counts_and_buckets(self, tmp_path: Path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        now = datetime.now(tz=timezone.utc)
        entries = [
            {"ts": now.isoformat(), "message": "1", "category": "api", "logger": "l1"},
            {"ts": now.isoformat(), "message": "2", "category": "api", "logger": "l1"},
            {"ts": now.isoformat(), "message": "3", "category": "cron", "logger": "l2"},
            {"ts": now.isoformat(), "message": "4", "logger": "l3"},
        ]
        with open(log_dir / LEDGER_FILENAME, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        stats = summarize(since_seconds=3600, log_dir=log_dir)
        assert stats["total"] == 4
        assert stats["by_category"] == {"api": 2, "cron": 1, "general": 1}
        assert stats["by_logger"]["l1"] == 2
        assert stats["ledger_bytes"] > 0


class TestLoggingChokepoint:
    """setup_logging must attach the ledger handler (ERROR+ only)."""

    def test_handler_installed_and_writes(self, tmp_path: Path):
        import hermes_logging

        hermes_logging.setup_logging(hermes_home=tmp_path, force=True)
        try:
            logger = logging.getLogger("hermes.test.ledger_choke")
            logger.warning("below floor — must NOT reach ledger")
            logger.error("choke point works")
            hermes_logging.flush_log_queue()
            path = tmp_path / "logs" / LEDGER_FILENAME
            assert path.exists()
            lines = [
                json.loads(l) for l in path.read_text().splitlines() if l.strip()
            ]
            msgs = [e["message"] for e in lines]
            assert "choke point works" in msgs
            assert not any("below floor" in m for m in msgs)
        finally:
            hermes_logging.flush_log_queue()
            hermes_logging._reset_queued_handlers()


class TestCmdErrors:
    @pytest.fixture()
    def seeded_ledger(self, tmp_path: Path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        now = datetime.now(tz=timezone.utc)
        entries = [
            {"ts": now.isoformat(), "message": "api boom", "category": "api",
             "logger": "api.lg"},
            {"ts": now.isoformat(), "message": "cron boom", "category": "cron",
             "logger": "cron.lg", "detail": "job_id=j1"},
        ]
        with open(log_dir / LEDGER_FILENAME, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

        import agent.error_ledger as el

        monkeypatch.setattr(el, "ledger_path", lambda log_dir=None: log_dir / LEDGER_FILENAME if log_dir else tmp_path / "logs" / LEDGER_FILENAME)
        # read_recent/summarize call the module-level ledger_path, so patching
        # the module attribute covers both.
        return log_dir

    def _args(self, **kw):
        base = dict(since=None, category=None, limit=20, stats=False, json=False)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_plain_output(self, seeded_ledger, capsys):
        from hermes_cli.main import cmd_errors

        rc = cmd_errors(self._args())
        out = capsys.readouterr().out
        assert rc == 0
        assert "api boom" in out
        assert "[cron]" in out

    def test_stats_output(self, seeded_ledger, capsys):
        from hermes_cli.main import cmd_errors

        rc = cmd_errors(self._args(stats=True))
        out = capsys.readouterr().out
        assert rc == 0
        assert "2 error(s)" in out
        assert "api: 1" in out

    def test_json_output_category_filter(self, seeded_ledger, capsys):
        from hermes_cli.main import cmd_errors

        rc = cmd_errors(self._args(category="cron", json=True))
        out = capsys.readouterr().out
        assert rc == 0
        lines = [json.loads(l) for l in out.strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["message"] == "cron boom"

    def test_invalid_since(self, seeded_ledger, capsys):
        from hermes_cli.main import cmd_errors

        rc = cmd_errors(self._args(since="banana"))
        assert rc == 1
        assert "Invalid --since" in capsys.readouterr().out
