"""Structured JSONL error ledger for Hermes Agent.

The ledger is the machine-readable counterpart to ``errors.log``: every
ERROR-or-above record lands as one JSON object per line in
``~/.hermes/logs/error-ledger.jsonl`` so ``hermes errors``, the doctor
"Recent Error History" section, and regression tooling can query failures
without parsing the human log format.

Design notes
------------
* **Stdlib only.** The audit considered structlog but a ``logging.Handler``
  subclass over the existing root-logger choke point gives identical JSONL
  output with zero new dependencies and inherits rotation + redaction
  behaviour from the logging stack already audited for secrets.
* **Atomic appends.** POSIX guarantees ``O_APPEND`` writes below PIPE_BUF
  are atomic, so concurrent CLI/gateway processes can share one ledger
  without a lock. Records are truncated below that bound (long tracebacks
  are clipped) so multi-process interleaving never corrupts a line.
* **Redaction.** Message and traceback text pass through the same
  ``redact_sensitive_text`` used for every other log file — secrets never
  land in the ledger.

Emit choke points (three, per the audit design):

1. The root-logger handler installed by ``hermes_logging.setup_logging``
   captures every ``logger.error`` / ``logger.exception`` call site-wide.
2. The API retry loop in ``agent/conversation_loop.py`` tags records with
   ``error_category="api"`` + provider/model/status context.
3. The cron failure finalizer in ``cron/scheduler.py`` tags records with
   ``error_category="cron"`` + job identity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Ledger file lives next to agent.log / errors.log.
LEDGER_FILENAME = "error-ledger.jsonl"

#: POSIX atomic-append bound (PIPE_BUF). Records serialized beyond this are
#: clipped so a concurrent writer can never observe a torn line.
_MAX_RECORD_BYTES = 4000

#: Fields callers may attach via ``extra=`` that survive into the ledger.
_KNOWN_EXTRA_FIELDS = ("error_category", "error_detail")

logger = logging.getLogger(__name__)


class ErrorLedgerFormatter(logging.Formatter):
    """Serialize a LogRecord as a single redacted JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        from agent.redact import redact_sensitive_text

        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        sid = getattr(record, "session_tag", "") or ""
        sid = sid.strip().strip("[]")
        if sid:
            payload["session"] = sid

        category = getattr(record, "error_category", None)
        if category:
            payload["category"] = str(category)
        detail = getattr(record, "error_detail", None)
        if detail:
            payload["detail"] = str(detail)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            tb = self.formatException(record.exc_info)
            payload["traceback"] = tb

        line = json.dumps(payload, ensure_ascii=False, default=str)
        line = redact_sensitive_text(line)

        # Atomic-append guard: clip oversized records (tracebacks dominate).
        # Progressive reduction — each step is re-checked so the invariant
        # "every emitted line fits _MAX_RECORD_BYTES" always holds, even with
        # a huge detail/message field.
        encoded = line.encode("utf-8", errors="replace")
        if len(encoded) > _MAX_RECORD_BYTES:
            payload["traceback"] = (payload.get("traceback") or "")[:800]
            payload["message"] = payload["message"][:400]
            payload["truncated"] = True
            line = redact_sensitive_text(
                json.dumps(payload, ensure_ascii=False, default=str)
            )
            if len(line.encode("utf-8", errors="replace")) > _MAX_RECORD_BYTES:
                payload.pop("detail", None)
                payload["traceback"] = (payload.get("traceback") or "")[:200]
                payload["message"] = payload["message"][:150]
                line = redact_sensitive_text(
                    json.dumps(payload, ensure_ascii=False, default=str)
                )
        return line


def ledger_path(log_dir: Optional[Path] = None) -> Path:
    """Resolve the ledger file path for a logs directory."""
    if log_dir is None:
        from hermes_cli.config import get_hermes_home

        log_dir = get_hermes_home() / "logs"
    return Path(log_dir) / LEDGER_FILENAME


def read_recent(
    *,
    limit: int = 20,
    since_seconds: Optional[float] = None,
    category: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read the newest ledger records, newest first.

    Tolerates torn/legacy lines (skipped silently — the ledger is an
    optimization over errors.log, never the source of truth).
    """
    path = ledger_path(log_dir)
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                if category and entry.get("category") != category:
                    continue
                if since_seconds is not None:
                    ts = entry.get("ts") or ""
                    try:
                        age = now_ts - datetime.fromisoformat(ts).timestamp()
                    except ValueError:
                        continue
                    if age > since_seconds:
                        continue
                records.append(entry)
    except OSError:
        return []

    records.reverse()  # file order is oldest-first; callers want newest-first
    return records[:limit]


def summarize(
    *,
    since_seconds: float = 86400.0,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Aggregate ledger stats for the doctor section and ``hermes errors --stats``."""
    path = ledger_path(log_dir)
    stats: Dict[str, Any] = {
        "total": 0,
        "by_category": {},
        "by_logger": {},
        "ledger_exists": path.exists(),
        "ledger_bytes": 0,
    }
    if not path.exists():
        return stats
    stats["ledger_bytes"] = path.stat().st_size

    now_ts = datetime.now(tz=timezone.utc).timestamp()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                ts = entry.get("ts") or ""
                try:
                    age = now_ts - datetime.fromisoformat(ts).timestamp()
                except ValueError:
                    continue
                if age > since_seconds:
                    continue
                stats["total"] += 1
                cat = entry.get("category") or "general"
                stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
                lg = entry.get("logger") or "?"
                stats["by_logger"][lg] = stats["by_logger"].get(lg, 0) + 1
    except OSError:
        pass
    return stats
