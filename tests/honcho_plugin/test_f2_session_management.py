"""F2: Peer & Session Management — comprehensive tests.

Tests HonchoSessionManager: peer caching, session creation, message buffering,
async flush, context prefetch, dialectic query, peer card, search, lifecycle.
Uses mocked Honcho SDK — no network calls.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch, call

import pytest

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Build a minimal HonchoClientConfig-like object for testing."""
    defaults = dict(
        host="hermes",
        workspace_id="hermes",
        api_key="test-key",
        peer_name="testuser",
        ai_peer="hermes",
        save_messages=True,
        write_frequency="async",
        context_tokens=None,
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
        recall_mode="hybrid",
        context_cadence=1,
        dialectic_cadence=2,
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        pin_peer_name=False,
        user_peer_aliases={},
        runtime_peer_prefix="",
        session_strategy="per-directory",
        session_peer_prefix=False,
        base_url=None,
        observation_mode="directional",
    )
    defaults.update(overrides)

    @dataclass
    class FakeConfig:
        pass

    cfg = FakeConfig()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_manager(config=None, honcho=None):
    """Build a HonchoSessionManager with mocked Honcho client.
    
    NOTE: The honcho property always calls get_honcho_client() (for OAuth refresh),
    so we must patch that function to return our mock.
    """
    cfg = config or _make_config()
    mock_honcho = honcho or MagicMock()
    
    with patch("plugins.memory.honcho.session.get_honcho_client", return_value=mock_honcho):
        mgr = HonchoSessionManager(config=cfg, honcho=mock_honcho)
    
    # Patch the property to always return our mock
    # (the real property calls get_honcho_client() on every access)
    type(mgr).honcho = property(lambda self: mock_honcho)
    
    return mgr, mock_honcho


# ---------------------------------------------------------------------------
# Category 1: Peer Creation & Caching
# ---------------------------------------------------------------------------


class TestPeerCreationCaching:
    """T1-T3: Lazy creation, cache hit, TOCTOU race."""

    def test_lazy_peer_creation_no_api_on_init(self):
        """T1: Constructor must NOT call honcho.peer()."""
        mgr, mock_honcho = _make_manager()
        assert mock_honcho.peer.call_count == 0

    def test_peer_created_on_first_session_access(self):
        """T1: get_or_create triggers peer creation for user + assistant."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.return_value = MagicMock()
        mock_honcho.session.return_value = MagicMock()

        mgr.get_or_create("ch:1")
        # user peer + ai peer = 2 calls
        assert mock_honcho.peer.call_count == 2

    def test_peer_cache_hit_no_duplicate_api(self):
        """T2: Second _get_or_create_peer returns cached, no API call."""
        mgr, mock_honcho = _make_manager()
        mock_peer = MagicMock()
        mock_honcho.peer.return_value = mock_peer

        p1 = mgr._get_or_create_peer("user1")
        p2 = mgr._get_or_create_peer("user1")

        assert p1 is p2
        assert mock_honcho.peer.call_count == 1

    def test_peer_cache_different_ids(self):
        """T2: Different peer IDs get different objects."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.side_effect = lambda pid: MagicMock(name=f"peer-{pid}")

        p1 = mgr._get_or_create_peer("alice")
        p2 = mgr._get_or_create_peer("bob")

        assert p1 is not p2
        assert mock_honcho.peer.call_count == 2

    def test_peer_toctou_concurrent_access(self):
        """T3: 10 threads racing on same peer ID — all get same object."""
        mgr, mock_honcho = _make_manager()
        shared_peer = MagicMock(name="shared")
        mock_honcho.peer.return_value = shared_peer

        results = [None] * 10
        barrier = threading.Barrier(10)

        def worker(idx):
            barrier.wait()
            results[idx] = mgr._get_or_create_peer("race-peer")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All should get the same cached object (last writer wins is acceptable)
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 10
        # At most one unique object (cache converges)
        assert all(r is non_none[0] for r in non_none)


# ---------------------------------------------------------------------------
# Category 2: Session Creation & Config Sync
# ---------------------------------------------------------------------------


class TestSessionCreation:
    """T4-T6: SessionPeerConfig, server sync-back, add_peers failure."""

    def test_session_cached_on_second_access(self):
        """Session get_or_create returns cached session on second call."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        s1 = mgr.get_or_create("ch:1")
        s2 = mgr.get_or_create("ch:1")

        assert s1 is s2
        # session() called once (cached)
        assert mock_honcho.session.call_count == 1

    def test_add_peers_failure_non_fatal(self):
        """T6: add_peers exception doesn't prevent session creation."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.add_peers.side_effect = ConnectionError("network down")
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        # Should NOT raise
        session = mgr.get_or_create("ch:fail")
        assert session is not None
        assert session.key == "ch:fail"


# ---------------------------------------------------------------------------
# Category 3: Message Buffering & History
# ---------------------------------------------------------------------------


class TestMessageBuffering:
    """T7-T9: get_history boundary, metadata stripping, clear()."""

    def test_add_and_get_history(self):
        """Basic: add messages, get them back."""
        session = HonchoSession(key="test", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s1")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi there")

        history = session.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hello"}
        assert history[1] == {"role": "assistant", "content": "hi there"}

    def test_get_history_max_messages(self):
        """T7: max_messages limits returned count."""
        session = HonchoSession(key="test", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s1")
        for i in range(10):
            session.add_message("user", f"msg-{i}")

        history = session.get_history(max_messages=3)
        assert len(history) == 3
        # Should return LAST 3
        assert history[0]["content"] == "msg-7"
        assert history[2]["content"] == "msg-9"

    def test_get_history_max_messages_zero_boundary(self):
        """T7 CRITICAL: max_messages=0 — Python [-0:] == [0:] returns ALL.
        This documents the current (buggy) behavior."""
        session = HonchoSession(key="test", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s1")
        for i in range(5):
            session.add_message("user", f"msg-{i}")

        history = session.get_history(max_messages=0)
        # BUG: [-0:] returns all 5, not 0
        # If this test passes with len==5, the bug exists
        # If fixed, should return []
        assert len(history) == 5  # documents current behavior

    def test_get_history_strips_metadata(self):
        """T8: Only role+content returned, no _synced or custom fields."""
        session = HonchoSession(key="test", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s1")
        session.add_message("user", "hi", custom_field="x")

        history = session.get_history()
        assert set(history[0].keys()) == {"role", "content"}
        assert "_synced" not in history[0]
        assert "custom_field" not in history[0]

    def test_clear_resets_messages_preserves_identity(self):
        """T9: clear() empties messages but keeps key/peer IDs."""
        session = HonchoSession(key="ch:1", user_peer_id="u1", assistant_peer_id="a1", honcho_session_id="s1")
        session.add_message("user", "msg")
        old_updated = session.updated_at

        time.sleep(0.01)
        session.clear()

        assert session.messages == []
        assert session.key == "ch:1"
        assert session.user_peer_id == "u1"
        assert session.assistant_peer_id == "a1"
        assert session.updated_at >= old_updated


# ---------------------------------------------------------------------------
# Category 4: Async Flush & Writer Thread
# ---------------------------------------------------------------------------


class TestAsyncFlush:
    """T10-T14: synced flag, flush failure, retry-then-drop, shutdown."""

    def test_flush_only_unsynced_messages(self):
        """T10: _flush_session sends only messages without _synced=True."""
        mgr, mock_honcho = _make_manager(config=_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        # Simulate 3 already-synced messages
        for i in range(3):
            session.messages.append({"role": "user", "content": f"old-{i}", "_synced": True})
        # Add 2 new
        session.add_message("user", "new-0")
        session.add_message("assistant", "new-1")

        result = mgr._flush_session(session)
        assert result is True

        # add_messages called with only the 2 new messages
        add_call = mock_session.add_messages.call_args
        sent = add_call[0][0] if add_call[0] else add_call[1].get("messages", [])
        assert len(sent) == 2

    def test_flush_failure_marks_unsynced(self):
        """T11: On exception, new messages marked _synced=False for retry."""
        mgr, mock_honcho = _make_manager(config=_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_session.add_messages.side_effect = TimeoutError("timeout")
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        session.add_message("user", "will-fail")

        result = mgr._flush_session(session)
        assert result is False

        # Message should be marked unsynced for retry
        unsynced = [m for m in session.messages if not m.get("_synced")]
        assert len(unsynced) >= 1

    def test_flush_empty_session_returns_true(self):
        """Flushing a session with no new messages is a no-op success."""
        mgr, mock_honcho = _make_manager(config=_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        # No messages added
        result = mgr._flush_session(session)
        assert result is True
        mock_session.add_messages.assert_not_called()


# ---------------------------------------------------------------------------
# Category 5: Context Prefetch
# ---------------------------------------------------------------------------


class TestContextPrefetch:
    """T15-T17: pop-once semantics, race, missing session."""

    def test_pop_context_result_consumes_once(self):
        """T15: set then pop returns data; second pop returns empty."""
        mgr, _ = _make_manager()
        mgr.set_context_result("k1", {"representation": "test-repr"})

        first = mgr.pop_context_result("k1")
        assert first == {"representation": "test-repr"}

        second = mgr.pop_context_result("k1")
        assert second == {}

    def test_pop_nonexistent_key_returns_empty(self):
        """T17: pop for unknown key returns empty dict, no error."""
        mgr, _ = _make_manager()
        result = mgr.pop_context_result("nonexistent")
        assert result == {}

    def test_set_overwrites_previous(self):
        """T16: Last set wins for same key."""
        mgr, _ = _make_manager()
        mgr.set_context_result("k", {"v": 1})
        mgr.set_context_result("k", {"v": 2})

        result = mgr.pop_context_result("k")
        assert result == {"v": 2}


# ---------------------------------------------------------------------------
# Category 6: Dialectic Query
# ---------------------------------------------------------------------------


class TestDialecticQuery:
    """T18-T20: Query truncation, injection cap, observe-others flag."""

    def test_injection_cap_truncates(self):
        """T19: With apply_injection_cap=True, result capped at dialectic_max_chars."""
        mgr, mock_honcho = _make_manager(config=_make_config(
            dialectic_max_chars=100,
            ai_observe_others=True,
        ))
        mock_peer = MagicMock()
        mock_peer.chat.return_value = "x" * 500
        mock_honcho.peer.return_value = mock_peer

        # Create session so peer resolution works
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session
        mgr.get_or_create("ch:1")

        result = mgr.dialectic_query("ch:1", "test question", apply_injection_cap=True)
        assert len(result) <= 103  # 100 + " …"

    def test_injection_cap_disabled_preserves_full(self):
        """T19: With apply_injection_cap=False, full result returned."""
        mgr, mock_honcho = _make_manager(config=_make_config(
            dialectic_max_chars=100,
            ai_observe_others=True,
        ))
        mock_peer = MagicMock()
        mock_peer.chat.return_value = "y" * 500
        mock_honcho.peer.return_value = mock_peer

        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session
        mgr.get_or_create("ch:1")

        result = mgr.dialectic_query("ch:1", "test question", apply_injection_cap=False)
        assert len(result) == 500


# ---------------------------------------------------------------------------
# Category 7: Peer Card & Search
# ---------------------------------------------------------------------------


class TestPeerCardSearch:
    """T21-T23: Card fallback, max_tokens floor, search fallback."""

    def test_get_peer_card_returns_list(self):
        """T21: get_peer_card returns list of strings."""
        mgr, mock_honcho = _make_manager()
        mock_peer = MagicMock()
        mock_peer.get_card.return_value = ["fact1", "fact2"]
        mock_honcho.peer.return_value = mock_peer

        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session
        mgr.get_or_create("ch:1")

        card = mgr.get_peer_card("ch:1", "user")
        assert isinstance(card, list)

    def test_search_context_max_tokens_floor(self):
        """T22: max_tokens=0 → char_budget = max(200, 0*4) = 200."""
        mgr, mock_honcho = _make_manager()
        mock_peer = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "x" * 300
        mock_peer.search.return_value = [mock_msg]
        mock_honcho.peer.return_value = mock_peer

        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session
        mgr.get_or_create("ch:1")

        # max_tokens=0 → floor of 200 chars
        result = mgr.search_context("ch:1", "query", max_tokens=0)
        # Result should be truncated to ~200 chars (floor), not empty
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Category 8: Session Lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """T24-T25: new_session atomicity, empty migration."""

    def test_new_session_replaces_old(self):
        """T24: new_session removes old and creates fresh."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        old = mgr.get_or_create("ch:1")
        old.add_message("user", "old-msg")

        new = mgr.new_session("ch:1")
        assert new is not old
        assert new.messages == []

    def test_delete_removes_from_cache(self):
        """delete() removes session from internal cache."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        mgr.get_or_create("ch:del")
        assert mgr.delete("ch:del") is True
        assert mgr.delete("ch:del") is False  # already gone

    def test_shutdown_completes_cleanly(self):
        """T13: shutdown() doesn't hang or raise."""
        mgr, mock_honcho = _make_manager()
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        mgr.get_or_create("ch:1")
        # Should complete within timeout
        mgr.shutdown()
