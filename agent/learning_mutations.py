"""User-initiated edit/delete for journey nodes (learned skills + memories).

Node ids (from ``agent.learning_graph``): skills → the skill name; memories →
``memory:<source>:<index>`` (``source`` = ``memory`` for MEMORY.md / ``profile``
for USER.md; ``index`` = position in the combined card list, MEMORY.md first).
Shared by CLI ``hermes journey``, the TUI ``/journey`` overlay and the desktop.
Deleting a skill *archives* it (``hermes curator restore`` recovers it);
deleting a memory rewrites its file.

Memory mutations run their locate→mutate→write span under the same
``MemoryStore._file_lock`` the memory tool uses, and re-resolve the target
entry by CONTENT inside the lock (W54-F028: a concurrent ``MemoryStore.add``
can never be clobbered by a journey write; W54-F034: the mutated entry is the
one the graph rendered, even if its index moved).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

_MEMORY_FILES = {"memory": "MEMORY.md", "profile": "USER.md"}


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
    """Resolve a memory node id to ``(path, source, content fingerprint)``.

    The fingerprint is the FULL chunk text of the entry the journey graph
    rendered for this node, captured read-only and unlocked. Under a
    concurrent writer the positional index may be stale by the time the
    mutation lands, so the mutation stage re-matches this content under the
    store's file lock instead of trusting ``gidx`` (W54-F034)."""
    from hermes_constants import get_hermes_home
    from agent.learning_graph import _memory_cards
    from tools.memory_tool import MemoryStore

    source, gidx = _parse_memory_id(node_id)
    path = get_hermes_home() / "memories" / _MEMORY_FILES[source]
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

    The file is re-read INSIDE the lock and the target entry re-resolved by
    content fingerprint: a concurrent ``MemoryStore`` writer (e.g. ``add``)
    that committed between our snapshot and our write can neither be
    clobbered by a stale write nor shift our mutation onto a different
    entry. Duplicate identical chunks resolve to the first, matching
    ``tools.memory_tool_store._find_unique_match``. Returns the mutated
    file's path."""
    from tools.memory_tool import MemoryStore

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
    _, _, body = _resolve_memory_identity(node_id)
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
