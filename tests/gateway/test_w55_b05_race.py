"""W55 B05 (W54-F035): cross-process lost-update on PairingStore load->modify->save.

The gateway process and the ``hermes pairing`` CLI each construct their own
PairingStore over the same JSON files. The store serializes mutations with a
threading.RLock only, which gives NO cross-process mutual exclusion, so two
processes can both load the same approved list, each add a different user, and
the last save wins -- one approval is silently lost.

This test drives that exact interleaving with two spawned processes:

- Each process approves a DIFFERENT pre-issued request id via
  ``approve_request`` (public API; mutates both ``-pending.json`` and
  ``-approved.json``).
- A file-based barrier is wired into the ``*-approved.json`` write: each
  process records that it has *loaded and mutated* (line appended AFTER the
  load, before the real write) and then waits until both lines are present.
  This guarantees BOTH processes hold stale copies before EITHER writes.

Without cross-process locking (the bug), both writes go through, the last one
clobbers the first, and the final JSON contains exactly one of the two users:
the assertion "both approvals present" fails deterministically.

With a cross-process flock held across load->mutate->save (the fix), the first
process holds the flock, the second blocks BEFORE loading, the barrier holder
times out (bounded poll -- the wait is skipped, not deadlocked), writes, and
releases; the second process then loads the FRESH file and both approvals
survive in the final JSON.
"""

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

import gateway.pairing as pairing_mod


SYNC_TIMEOUT_SECONDS = 8


def _w55_b05_approve_child(pairing_dir: str, sync_path: str, platform: str,
                           request_id: str, user: str) -> dict:
    """One spawned process: one approve_request, synchronized at the approved write."""
    import gateway.pairing as mod  # fresh module state in the spawned interpreter

    mod.PAIRING_DIR = Path(pairing_dir)
    sync_file = Path(sync_path)
    expected_suffix = f"-approved.json"

    real_secure_write = mod._secure_write
    state = {"synced": False}

    def _sync_secure_write(path: Path, data: str) -> None:
        # Only the approved-list write participates in the barrier; every other
        # write (pending, rate limits) passes straight through.
        if not state["synced"] and str(path).endswith(expected_suffix):
            state["synced"] = True
            with open(sync_file, "a", encoding="utf-8") as f:
                f.write(f"{os.getpid()}:{user}\n")
                f.flush()
            deadline = time.monotonic() + SYNC_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                try:
                    lines = [l for l in sync_file.read_text(encoding="utf-8").splitlines() if l]
                except OSError:
                    lines = []
                if len(lines) >= 2:
                    # Both processes have loaded (stale) state; proceed to a racing write
                    # on the UNFIXED store, or a serialized one under flock.
                    break
                time.sleep(0.02)
        real_secure_write(path, data)

    mod._secure_write = _sync_secure_write

    store = mod.PairingStore()
    result = store.approve_request(platform, request_id)
    assert result is not None, f"approve_request returned None for {user}"
    return result


def test_two_process_approvals_not_lost(tmp_path):
    """Concurrent approvals from two PROCESSES must both survive (no lost update)."""
    pairing_dir = tmp_path / "pairing"
    pairing_mod.PAIRING_DIR = pairing_dir
    store = pairing_mod.PairingStore()
    platform = "telegram"
    users = ["alice", "bob"]

    for user in users:
        assert store.generate_code(platform, user, user) is not None
    pending = {p["user_id"]: p["request_id"] for p in store.list_pending(platform)}
    assert len(pending) == 2, f"expected 2 pending requests, got {pending}"

    sync_file = tmp_path / "sync.txt"
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_w55_b05_approve_child,
            args=(str(pairing_dir), str(sync_file), platform, pending[user], user),
        )
        for user in users
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
    assert [p.exitcode for p in procs] == [0, 0], f"child exit codes: {[p.exitcode for p in procs]}"

    approved_path = pairing_dir / f"{platform}-approved.json"
    approved = json.loads(approved_path.read_text(encoding="utf-8"))
    for user in users:
        assert user in approved, f"approval for {user!r} lost: final approved = {approved}"


if __name__ == "__main__":  # pragma: no cover - direct run convenience
    pytest.main([__file__, "-q"])