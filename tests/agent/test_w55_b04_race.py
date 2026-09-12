"""Race contracts for journey node edit/delete (battery B04: W54-F028 / W54-F034).

Thread A mutates a memory node through ``agent.learning_mutations`` while thread B
adds entries through the real ``MemoryStore`` against the SAME store file.

Two findings are pinned:

* **F028 — no lost update.** ``learning_mutations`` performed its
  locate→mutate→write span without ``MemoryStore._file_lock``, so an ``add``
  committed by a concurrent writer in that window is silently clobbered by the
  stale-snapshot write. Both the mutation AND the add must survive.
* **F034 — content identity, not position.** Node ids are positional
  (``memory:<source>:<gidx>``) page data; under a concurrent writer the entry
  to mutate must be re-resolved by CONTENT (fingerprint captured before the
  lock, re-matched under it). Assertions are made purely on entry content —
  never on indices — so a mutation that landed on the wrong entry fails.

Determinism without sleeps: the F028 tests are choreographed with events. An
event fires the moment the mutation has READ its snapshot, and a second event
is set only after the racing adds have fully committed. Pre-fix this forces
the add inside the locate→write window (guaranteed clobber → RED). Post-fix
the snapshot read happens pre-lock and the write is re-resolved under the
store's existing flock, so the same choreography serializes cleanly and no
lost update is possible (deterministic GREEN, no deadlock — the second event
is set after the add completes, never while holding the store lock).
"""

from __future__ import annotations

import threading

import pytest

from agent import learning_mutations as lm
from hermes_constants import get_hermes_home
from tools.memory_tool import MemoryStore


def _memory_path(name: str = "MEMORY.md"):
    return get_hermes_home() / "memories" / name


@pytest.fixture
def store_files():
    """MEMORY.md with a known victim at global index 1; USER.md with two cards."""
    home = get_hermes_home()
    (home / "memories").mkdir(parents=True, exist_ok=True)
    mm = home / "memories" / "MEMORY.md"
    us = home / "memories" / "USER.md"
    mm.write_text("alpha start\n§\nvictim original", encoding="utf-8")
    us.write_text("up one\n§\nup two", encoding="utf-8")
    return mm, us


def _fresh_store() -> MemoryStore:
    """Huge budgets so a racing ``add`` always succeeds (never a budget refusal)."""
    return MemoryStore(memory_char_limit=5_000_000, user_char_limit=5_000_000)


def _install_snapshot_gate(monkeypatch, read_done: threading.Event,
                           add_committed: threading.Event) -> None:
    """Choreograph thread A's snapshot read.

    ``agent.learning_graph._memory_cards`` is the graph snapshot both the
    pre-fix ``_locate_memory`` and the fixed identity resolver read before
    writing. Fire ``read_done`` the instant that snapshot is taken, and block
    until ``add_committed`` — which the racing add thread sets only after its
    writes hit the file. The wrapper is inert on threads that do not call
    ``_memory_cards``, i.e. it never touches the racing ``MemoryStore.add``.
    """
    import agent.learning_graph as lg

    real_cards = lg._memory_cards

    def gated_cards():
        result = real_cards()
        if threading.current_thread().name == "mutator-A":
            read_done.set()
            add_committed.wait()
        return result

    monkeypatch.setattr(lg, "_memory_cards", gated_cards)


def _run_mutator_thread(fn, barrier: threading.Barrier, read_done: threading.Event,
                        outcome: list) -> threading.Thread:
    """Start thread A behind the barrier; never strand the main thread if the
    mutation fails before reaching its snapshot read."""

    def worker():
        try:
            barrier.wait()
            result = fn()
            if isinstance(result, dict):
                outcome.append(result)
        except Exception as exc:  # surfaced via the final-state assertions
            outcome.append(repr(exc))
        finally:
            read_done.set()

    thread = threading.Thread(target=worker, name="mutator-A")
    thread.start()
    return thread


# ── F028: no lost update ────────────────────────────────────────────────────

def test_f028_delete_and_concurrent_add_both_survive(store_files, monkeypatch):
    """A journey delete must not clobber entries a concurrent add committed.

    Choreography: A reads its snapshot → racing adds commit on MEMORY.md →
    A writes its stale snapshot. Pre-fix the write silently drops the adds.
    """
    mm, _ = store_files
    read_done, add_committed = threading.Event(), threading.Event()
    _install_snapshot_gate(monkeypatch, read_done, add_committed)
    barrier = threading.Barrier(2)
    outcome: list = []
    thread = _run_mutator_thread(
        lambda: lm.delete_node("memory:memory:1"), barrier, read_done, outcome)

    barrier.wait()
    read_done.wait()
    store = _fresh_store()
    for i in range(4):
        assert store.add("memory", f"concurrent note alpha {i}")["success"]
    add_committed.set()
    thread.join(timeout=30)
    assert not thread.is_alive(), "mutator thread hung"

    entries = MemoryStore._read_file(mm)
    # F028: the racing adds survive… (assert by content)
    for i in range(4):
        assert f"concurrent note alpha {i}" in entries, "racing add was lost (F028)"
    # …and the mutation landed too, on the right entry (assert by content).
    assert "victim original" not in entries, "delete did not land (F028)"
    assert "alpha start" in entries, "bystander entry was mutated"


def test_f028_edit_and_concurrent_add_both_survive(store_files, monkeypatch):
    """A journey edit must not resurrect stale state over a committed add."""
    mm, _ = store_files
    read_done, add_committed = threading.Event(), threading.Event()
    _install_snapshot_gate(monkeypatch, read_done, add_committed)
    barrier = threading.Barrier(2)
    outcome: list = []
    thread = _run_mutator_thread(
        lambda: lm.edit_node("memory:memory:1", "victim edited content"),
        barrier, read_done, outcome)

    barrier.wait()
    read_done.wait()
    store = _fresh_store()
    for i in range(4):
        assert store.add("memory", f"concurrent note beta {i}")["success"]
    add_committed.set()
    thread.join(timeout=30)
    assert not thread.is_alive(), "mutator thread hung"

    entries = MemoryStore._read_file(mm)
    for i in range(4):
        assert f"concurrent note beta {i}" in entries, "racing add was lost (F028)"
    assert "victim edited content" in entries, "edit did not land (F028)"
    assert "victim original" not in entries, "edit landed on the wrong content"
    assert "alpha start" in entries, "bystander entry was mutated"


# ── F034: content identity under shifted indices ───────────────────────────

def test_f034_right_entry_mutated_across_shifted_indices(store_files):
    """Rounds of edit-vs-add contention: the mutated entry is always the one
    chosen by content identity, never by a position that concurrent adds can
    shift. Assert by content only."""
    mm, _ = store_files
    rounds, adds_per_round = 8, 5
    store = _fresh_store()
    for r in range(rounds):
        victim = f"victim round {r}"
        MemoryStore._write_file(mm, MemoryStore._read_file(mm) + [victim, f"neighbor {r}"])
        entries = MemoryStore._read_file(mm)
        gidx = entries.index(victim)  # the positional id the journey graph renders
        barrier = threading.Barrier(2)
        outcome: list = []
        thread = _run_mutator_thread(
            lambda: lm.edit_node(f"memory:memory:{gidx}", f"{victim} EDITED"),
            barrier, threading.Event(), outcome)
        barrier.wait()
        for j in range(adds_per_round):
            assert store.add("memory", f"concurrent add {r}-{j}")["success"]
        thread.join(timeout=30)
        assert not thread.is_alive(), "mutator thread hung"

    entries = MemoryStore._read_file(mm)
    for r in range(rounds):
        # F034: the edit hit the intended entry (by content), not a shifted index.
        assert f"victim round {r} EDITED" in entries, f"round {r}: edit hit wrong entry"
        assert f"victim round {r}\n" not in "\n".join(entries) + "\n", \
            f"round {r}: victim left unedited"
        for j in range(adds_per_round):
            assert f"concurrent add {r}-{j}" in entries, \
                f"round {r}: racing add lost (F028/F034)"
        assert f"neighbor {r}" in entries, f"round {r}: bystander mutated"