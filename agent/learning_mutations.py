"""User-initiated edit/delete for journey nodes (learned skills + memories).

Node ids (from ``agent.learning_graph``): skills → the skill name; memories →
``memory:<source>:<index>`` (``source`` = ``memory`` for MEMORY.md / ``profile``
for USER.md; ``index`` = position in the combined card list, MEMORY.md first).
Shared by CLI ``hermes journey``, the TUI ``/journey`` overlay and the desktop.
Deleting a skill *archives* it (``hermes curator restore`` recovers it);
deleting a memory rewrites its file.

Memory mutations run their mutate→write span under the same
``MemoryStore._file_lock`` the memory tool uses, so a concurrent
``MemoryStore`` write can never be clobbered by a journey write (W54-F028).

Node identity is a CONTENT FINGERPRINT captured at PREFILL time (W54-F034)
and matched (never re-derived positionally) inside the lock at commit:
``node_detail`` — the edit prefill — records the exact chunk text it rendered,
and the commit re-matches that chunk by full-text equality under the lock. A
concurrent write shifting indices during the prefill→commit window can
neither move the mutation onto a different entry nor delete/rewrite the new
occupant: zero matches raise an explicit stale error ("memory node changed
since prefill — refresh the graph"), surfaced as ``ok=False``. Duplicate
identical chunks (two nodes whose content is byte-identical) resolve
deterministically to the FIRST match in file order, identical to
``tools.memory_tool_store._find_unique_match`` — defined, documented
behavior: the prefill cannot distinguish identical bodies, so position is
only a tie-breaker supplied by the file itself.

Ids committed WITHOUT a same-process prefill (direct API calls, e.g. the
web PUT or a scripted ``edit_node``/``delete_node``) keep the positional
fallback: the id still resolves positionally and the entry is re-read and
re-matched by content under the lock (same semantics as the round-1 fix), so
positional edits/deletes keep working and F028 still holds for them.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

_MEMORY_FILES = {"memory": "MEMORY.md", "profile": "USER.md"}


# Prefill registry: node_id → (store path, chunk text rendered at prefill).
# Captured by ``node_detail`` (the edit prefill), consumed once by the commit
# under the same process. Capped FIFO so a long-lived process cannot grow it
# without bound.
_PREFILLS: dict[str, tuple[Path, str]] = {}
_PREFILLS_GUARD = threading.Lock()
_PREFILLS_MAX = 256


def _register_prefill(node_id: str, path: Path, chunk: str) -> None:
    """Capture the content fingerprint (full chunk text) of the entry the
    graph rendered for ``node_id`` at prefill time — W54-F034: the commit
    re-matches THIS text under the lock, never a re-derived position."""
    with _PREFILLS_GUARD:
        if node_id not in _PREFILLS and len(_PREFILLS) >= _PREFILLS_MAX:
            _PREFILLS.pop(next(iter(_PREFILLS)), None)
        _PREFILLS[node_id] = (path, chunk)


def _take_prefill(node_id: str) -> tuple[Path, str] | None:
    """Pop the prefill-time fingerprint for ``node_id`` (one-shot: each
    commit consumes its own prefill). ``None`` when the id was passed to the
    mutation API without an in-process ``node_detail`` prefill."""
    with _PREFILLS_GUARD:
        return _PREFILLS.pop(node_id, None)


def parse_node_kind(node_id: str) -> str:
    return "memory" if node_id.startswith("memory:") else "skill"


def _parse_memory_id(node_id: str) -> tuple[str, int]:
    """``memory:<source>:<index>`` → (source, global_index)."""
    parts = node_id.split(":", 2)
    try:
        if len(parts) != 3 or parts[0] != "memory" or parts[1] not in _MEMORY_FILES:
            raise ValueError
        return parts[1], int(parts[2])
    except ValueError as exc:
        raise ValueError(f"bad memory node id: {node_id!r}") from exc


def _resolve_memory_identity(node_id: str) -> tuple[Path, str, str]:
    """Resolve a memory node id to ``(path, source, chunk text)``.

    The path comes from ``MemoryStore._path_for`` (structural identity with
    the store's lock, not a duplicated derivation), the chunk is the entry
    the journey graph rendered for this node, captured read-only and
    unlocked. Under a concurrent writer the positional index may be stale by
    the time a mutation lands, so mutations re-match this content under the
    store's file lock instead of trusting ``gidx`` (W54-F034)."""
    from agent.learning_graph import _memory_cards
    from tools.memory_tool import MemoryStore

    source, gidx = _parse_memory_id(node_id)
    path = MemoryStore._path_for("user" if source == "profile" else "memory")
    if not path.exists():
        raise ValueError(f"{path.name} not found")
    cards = _memory_cards()
    if not 0 <= gidx < len(cards):
        raise IndexError(f"memory index {gidx} out of range")
    if cards[gidx].get("source") != source:
        raise ValueError("memory node id is stale — refresh the graph")
    chunks = MemoryStore._read_file(path)
    mem_count = sum(1 for c in cards if c.get("source") == "memory")
    local = gidx if source == "memory" else gidx - mem_count
    if not 0 <= local < len(chunks):
        raise ValueError("memory node id is stale — refresh the graph")
    return path, source, chunks[local]


def _mutate_memory_locked(node_id: str, mutate: Callable[[list[str], int], None]) -> Path:
    """Locate→mutate→write under the store's file lock (W54-F028).

    The target entry is resolved by the CONTENT FINGERPRINT captured at
    PREFILL time (``node_detail``) and re-matched by full-text equality
    INSIDE the lock — never derived positionally at commit (W54-F034). A
    concurrent ``MemoryStore`` write that lands between prefill and commit
    can neither be clobbered nor shift the mutation onto a different entry:

    * prefilled ids: zero matches raise an explicit ``ValueError`` (``memory
      node changed since prefill — refresh the graph``) — surfaced as
      ``ok=False``; duplicate identical chunks resolve deterministically to
      the FIRST match in file order, matching
      ``tools.memory_tool_store._find_unique_match``;
    * ids without a same-process prefill (direct API calls): positional
      fallback with the same under-lock re-read + content re-match, so
      positional edits/deletes keep working (F028 still holds).

    Returns the mutated file's path."""
    from tools.memory_tool import MemoryStore

    prefill = _take_prefill(node_id)
    if prefill is not None:
        path, chunk = prefill
        with MemoryStore._file_lock(path):
            chunks = MemoryStore._read_file(path)
            matches = [i for i, c in enumerate(chunks) if c == chunk]
            if not matches:
                raise ValueError("memory node changed since prefill — refresh the graph")
            mutate(chunks, matches[0])
            _write_memory(path, chunks)
        return path
    path, _, fingerprint = _resolve_memory_identity(node_id)
    with MemoryStore._file_lock(path):
        chunks = MemoryStore._read_file(path)
        matches = [i for i, chunk in enumerate(chunks) if chunk == fingerprint]
        if not matches:
            raise ValueError("memory node id is stale — refresh the graph")
        mutate(chunks, matches[0])
        _write_memory(path, chunks)
    return path


def _write_memory(path: Path, chunks: list[str]) -> None:
    """Atomic temp-file + rename via the memory tool, so a concurrent reader
    never sees a half-written file (and the §-join stays single-sourced)."""
    from tools.memory_tool import MemoryStore
    MemoryStore._write_file(path, [c.strip() for c in chunks if c.strip()])


def _clear_skill_cache() -> None:
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass


def _dispatch(node_id: str, memory_fn: Callable, skill_fn: Callable, *args) -> dict[str, Any]:
    try:
        return (memory_fn if parse_node_kind(node_id) == "memory" else skill_fn)(node_id, *args)
    except (ValueError, IndexError) as exc:
        return {"ok": False, "message": str(exc)}


# ── Inspect (edit prefill) ──────────────────────────────────────────────────

def node_detail(node_id: str) -> dict[str, Any]:
    """Current content for an edit prefill. ``content`` is the full SKILL.md
    (skills) or the raw memory chunk (memories)."""
    return _dispatch(node_id, _memory_detail, _skill_detail)


def _memory_detail(node_id: str) -> dict[str, Any]:
    path, _, body = _resolve_memory_identity(node_id)
    # Prefill capture (W54-F034): bind this id to the chunk rendered HERE so
    # the later commit re-matches THIS content under the lock instead of
    # re-deriving it positionally once the file may have moved on.
    _register_prefill(node_id, path, body)
    body = body.strip()
    return {"ok": True, "kind": "memory", "id": node_id, "label": body.splitlines()[0][:80], "content": body}


def _skill_detail(node_id: str) -> dict[str, Any]:
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(node_id)
    if not found:
        return {"ok": False, "message": f"skill '{node_id}' not found"}
    skill_md = Path(found["path"]) / "SKILL.md"
    if not skill_md.exists():
        return {"ok": False, "message": f"SKILL.md missing for '{node_id}'"}
    return {"ok": True, "kind": "skill", "id": node_id, "label": node_id, "content": skill_md.read_text(encoding="utf-8")}


# ── Delete ──────────────────────────────────────────────────────────────────

def delete_node(node_id: str) -> dict[str, Any]:
    return _dispatch(node_id, _delete_memory, _delete_skill)


def _delete_skill(name: str) -> dict[str, Any]:
    from tools import skill_usage
    # Pin must be respected by autonomous maintenance. The curator already skips pinned skills from every
    # auto-transition; the background review fork is the same kind of autonomous, no-user-present actor, so
    # it must not write to a pinned skill either (issue #25839). This is stricter than the foreground
    # ``_pinned_guard`` (which only blocks deletion) precisely because there is no user in the loop to
    # consent to an edit here.
    if skill_usage.get_record(name).get("pinned"):
        return {"ok": False, "message": f"'{name}' is pinned — unpin it first (hermes curator unpin {name})"}
    ok, message = skill_usage.archive_skill(name)
    if ok:
        _clear_skill_cache()
    return {"ok": ok, "message": f"archived '{name}' — restore with: hermes curator restore {name}" if ok else message}


def _delete_memory(node_id: str) -> dict[str, Any]:
    path = _mutate_memory_locked(node_id, lambda chunks, idx: chunks.__delitem__(idx))
    return {"ok": True, "message": f"deleted memory from {path.name}"}


# ── Edit ────────────────────────────────────────────────────────────────────

def edit_node(node_id: str, content: str) -> dict[str, Any]:
    return _dispatch(node_id, _edit_memory, _edit_skill, content)


def _edit_skill(name: str, content: str) -> dict[str, Any]:
    from tools.skill_manager_tool import _edit_skill as _do_edit
    result = _do_edit(name, content)
    if result.get("success"):
        _clear_skill_cache()
        return {"ok": True, "message": f"updated '{name}'"}
    return {"ok": False, "message": result.get("error", "edit failed")}


def _edit_memory(node_id: str, content: str) -> dict[str, Any]:
    _parse_memory_id(node_id)  # id errors win over the empty-body message
    body = content.strip()
    if not body:
        return {"ok": False, "message": "empty memory — use delete to remove it"}
    path = _mutate_memory_locked(
        node_id, lambda chunks, idx: chunks.__setitem__(idx, body))
    return {"ok": True, "message": f"updated memory in {path.name}"}
