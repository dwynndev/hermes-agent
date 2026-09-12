"""Wave 55 battery B03 race regression tests for ``gateway/status.py``.

Each test pins one W54 finding with a deterministic forced interleaving:

- W54-F016  ``write_runtime_status`` unlocked read-merge-write loses the
  interleaved update (two writer threads).
- W54-F017  ``acquire_scoped_lock`` empty-file branch bare-unlinks instead of
  atomically claiming, so two racing starters can both win (claim race).
- W54-F018  ``acquire_gateway_runtime_lock`` EACCES unlink-recreate leaves the
  winner holding flock on an inode the canonical path no longer names.
- W54-F019  ``gateway-starts.log`` storm ledger unlocked RMW drops a
  concurrent start.
- W54-F021  storm ledger trim rewrites through a SHARED tmp path (two
  concurrent trimmers corrupt/lose the interleaved entry).
- W54-F022  ``terminate_pid(force=False)`` ignores a provided
  ``expected_start_time`` (recycled-PID SIGTERM).
- W54-F023  ``is_gateway_runtime_lock_active`` probe UNLINKS a root-owned
  lock file on PermissionError (probes must not mutate).

Thread races (F016/F019/F021) rendezvous both workers on the read seam so the
two reads provably see the same pre-state; the cross-process claim races
(F017/F018) use the same single-actor racing choreography the existing
``test_acquire_scoped_lock_race_second_acquirer_loses`` uses for the stale
branch, extended to the empty-file branch.
"""

import contextlib
import json
import os
import threading
from pathlib import Path

import pytest

from gateway import status


_READ_RENDEZVOUS_TIMEOUT_S = 3.0


def _run_two_threads(fn_a, fn_b, names=("b03-a", "b03-b")) -> None:
    """Run two callables concurrently; re-raise any worker exception after joining."""
    errors: list = []

    def _guard(fn):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised after join
            errors.append(exc)

    threads = [threading.Thread(target=_guard, args=(fn,), name=name) for fn, name in ((fn_a, names[0]), (fn_b, names[1]))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "worker thread did not finish"
    if errors:
        raise errors[0]


class TestWriteRuntimeStatusLostUpdate:
    """W54-F016: two writer threads must merge, not clobber."""

    def test_concurrent_writes_merge_both_updates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        status_path = status._get_runtime_status_path()

        real_read = status._read_json_file
        read_barrier = threading.Barrier(2)

        def rendezvous_read(path, *args, **kwargs):
            # Both writers must have read the SAME pre-state before either writes:
            # that is the interleaving that makes an unlocked merge lose an update.
            if str(path) == str(status_path):
                try:
                    read_barrier.wait(timeout=_READ_RENDEZVOUS_TIMEOUT_S)
                except threading.BrokenBarrierError:
                    pass  # serialized writer: the peer is parked on the RMW lock
            return real_read(path, *args, **kwargs)

        monkeypatch.setattr(status, "_read_json_file", rendezvous_read)

        _run_two_threads(
            lambda: status.write_runtime_status(gateway_state="thread-a"),
            lambda: status.write_runtime_status(active_agents=7),
        )

        record = status.read_runtime_status()
        assert record is not None
        assert record["gateway_state"] == "thread-a"
        assert record["active_agents"] == 7


class TestScopedLockEmptyFileClaimRace:
    """W54-F017: the empty-file cleanup must be an atomic claim, not a bare unlink."""

    def test_empty_file_race_second_acquirer_loses(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Crash between O_EXCL create and json.dump: an EMPTY lock file.
        lock_path.write_text("", encoding="utf-8")

        winner_record = {
            "pid": 424242,
            "start_time": 456,
            "kind": "hermes-gateway",
            "scope": "telegram-bot-token",
        }
        real_replace = os.replace

        def racing_replace(src, dst, *args, **kwargs):
            if str(src) == str(lock_path):
                # Winner completed claim + O_EXCL re-create between our emptiness
                # check and our own claim attempt; we must fall through to O_EXCL
                # and LOSE — a bare unlink would instead destroy the winner's file.
                lock_path.write_text(json.dumps(winner_record), encoding="utf-8")
                raise FileNotFoundError(2, "No such file or directory", str(src))
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(status.os, "replace", racing_replace)

        acquired, existing = status.acquire_scoped_lock(
            "telegram-bot-token", "secret", metadata={"platform": "telegram"}
        )

        assert acquired is False
        assert existing is not None
        assert existing["pid"] == 424242
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == 424242

    def test_empty_file_plain_flow_still_wins(self, monkeypatch, tmp_path):
        # Non-racing invariant: a crashed-create leftover is still claimed.
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        lock_path = tmp_path / "locks" / "telegram-bot-token-2bb80d537b1da3e3.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("", encoding="utf-8")

        acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "secret")

        assert acquired is True
        assert existing is None
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


class TestGatewayRuntimeLockRecreateInodeRace:
    """W54-F018: after the EACCES unlink-recreate branch, a won flock must still
    name the canonical path's inode."""

    def test_eacces_recreate_loses_when_path_names_other_inode(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        lock_path = status._get_gateway_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("stale", encoding="utf-8")

        real_open = open

        def deny_first_open(path, *args, **kwargs):
            # Root-owned: opening fails while the ORIGINAL stale file is on disk;
            # after unlink, the fresh file the retry creates opens fine.
            if (
                str(path) == str(lock_path)
                and lock_path.exists()
                and lock_path.read_text(encoding="utf-8") == "stale"
            ):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_first_open)

        # A rival gateway takes the same EACCES recovery branch after OUR recreate:
        # the inode we then flock is no longer the one the canonical path names.
        real_try_lock = status._try_acquire_file_lock
        rival_path = lock_path.with_name("gateway.lock.rival")
        rival_state = {"done": False}

        def rival_recreates_before_lock(handle):
            if not rival_state["done"]:
                rival_state["done"] = True
                os.replace(lock_path, rival_path)
                lock_path.write_text("rival", encoding="utf-8")
            return real_try_lock(handle)

        monkeypatch.setattr(status, "_try_acquire_file_lock", rival_recreates_before_lock)

        try:
            won = status.acquire_gateway_runtime_lock()
        finally:
            status.release_gateway_runtime_lock()
            with contextlib.suppress(OSError):
                rival_path.unlink()
        assert won is False


class TestStormLedgerConcurrency:
    """W54-F019 + W54-F021: concurrent starts must all be recorded."""

    def _rendezvous_ledger_reads(self, monkeypatch):
        real_read_text = Path.read_text
        read_barrier = threading.Barrier(2)

        def rendezvous_read(self, *args, **kwargs):
            if self.name == "gateway-starts.log":
                try:
                    read_barrier.wait(timeout=_READ_RENDEZVOUS_TIMEOUT_S)
                except threading.BrokenBarrierError:
                    pass
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", rendezvous_read)
        return real_read_text

    def test_concurrent_starts_keep_both_entries(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        ledger = tmp_path / "gateway-starts.log"
        ledger.write_text("1600000000.0\n", encoding="utf-8")
        real_read_text = self._rendezvous_ledger_reads(monkeypatch)

        _run_two_threads(
            lambda: status.record_start_and_check_storm(),
            status.record_start_and_check_storm,
        )

        lines = [line for line in real_read_text(ledger).splitlines() if line.strip()]
        assert len(lines) == 3, f"both starts must survive the ledger RMW, got {lines!r}"

    def test_concurrent_trims_keep_both_starts_and_stay_bounded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        ledger = tmp_path / "gateway-starts.log"
        ledger.write_text("".join(f"{3.0 + i}\n" for i in range(45)), encoding="utf-8")
        real_read_text = self._rendezvous_ledger_reads(monkeypatch)

        # Distinct per-thread timestamps so each concurrent start is identifiable.
        stamps = {"b03-a": 1000.0, "b03-b": 2000.0}
        real_datetime = status.datetime

        class _StampDatetime:
            @classmethod
            def now(cls, tz=None):
                stamp = stamps.get(threading.current_thread().name)
                if stamp is not None:
                    return real_datetime.fromtimestamp(stamp, tz)
                return real_datetime.now(tz)

        monkeypatch.setattr(status, "datetime", _StampDatetime)

        _run_two_threads(
            lambda: status.record_start_and_check_storm(),
            status.record_start_and_check_storm,
        )

        lines = [line.strip() for line in real_read_text(ledger).splitlines() if line.strip()]
        assert "1000.0" in lines and "2000.0" in lines
        assert len(lines) <= 40, "trimmed ledger must stay bounded"


class TestTerminatePidNonForceGuard:
    """W54-F022: the expected_start_time fingerprint guard must refuse a
    recycled-PID SIGTERM too, not only force kills."""

    def test_nonforce_sigterm_refuses_recycled_pid(self, monkeypatch):
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 999)
        kills: list = []
        monkeypatch.setattr(status.os, "kill", lambda pid, sig: kills.append((pid, sig)))

        with pytest.raises(OSError, match="refusing to kill"):
            status.terminate_pid(4242, force=False, expected_start_time=123)

        assert kills == [], "a mismatched fingerprint must never be signalled"


class TestRuntimeLockProbeDoesNotMutate:
    """W54-F023: liveness probes must report, not unlink."""

    def test_probe_reports_inactive_without_unlinking(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("stale", encoding="utf-8")

        real_open = open

        def deny_write(path, *args, **kwargs):
            if str(path) == str(lock_path):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_write)

        assert status.is_gateway_runtime_lock_active(lock_path) is False
        assert lock_path.exists(), "probing liveness must not unlink the lock file"
        assert lock_path.read_text(encoding="utf-8") == "stale"