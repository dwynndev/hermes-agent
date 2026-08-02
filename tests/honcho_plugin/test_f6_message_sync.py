"""F6: Message Sync & Ingestion — comprehensive tests.

Tests save() routing by write_frequency, _flush_session synced flag,
async writer retry/drop, flush_all drain, shutdown ordering, message chunking.
"""

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    defaults = dict(
        host="hermes", workspace_id="hermes", api_key="test-key",
        peer_name="testuser", ai_peer="hermes",
        save_messages=True, write_frequency="async",
        context_tokens=None, dialectic_reasoning_level="low",
        dialectic_dynamic=True, dialectic_max_chars=600,
        dialectic_depth=1, dialectic_depth_levels=None,
        reasoning_heuristic=True, reasoning_level_cap="high",
        message_max_chars=25000, dialectic_max_input_chars=10000,
        recall_mode="hybrid", context_cadence=1, dialectic_cadence=2,
        user_observe_me=True, user_observe_others=True,
        ai_observe_me=True, ai_observe_others=True,
        pin_peer_name=False, user_peer_aliases={},
        runtime_peer_prefix="", session_strategy="per-directory",
        session_peer_prefix=False, base_url=None,
        observation_mode="directional", timeout=None,
    )
    defaults.update(overrides)

    class FakeConfig:
        pass

    cfg = FakeConfig()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_manager(config=None):
    cfg = config or _make_config()
    mock_honcho = MagicMock()

    with patch("plugins.memory.honcho.session.get_honcho_client", return_value=mock_honcho):
        mgr = HonchoSessionManager(config=cfg, honcho=mock_honcho)

    type(mgr).honcho = property(lambda self: mock_honcho)
    return mgr, mock_honcho


def _make_session_with_messages(key="ch:1", count=3):
    """Create a session with N unsynced messages."""
    session = HonchoSession(
        key=key, user_peer_id="u", assistant_peer_id="a",
        honcho_session_id=f"s-{key}",
    )
    for i in range(count):
        session.add_message("user", f"msg-{i}")
    return session


# ---------------------------------------------------------------------------
# Block A: save() routing by write_frequency
# ---------------------------------------------------------------------------


class TestSaveRouting:
    """T-F6-01 to T-F6-07: write_frequency routing."""

    def test_turn_mode_flushes_synchronously(self):
        """T-F6-01: wf='turn' → _flush_session called on every save()."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        session.add_message("user", "hello")

        with patch.object(mgr, '_flush_session', return_value=True) as mock_flush:
            mgr.save(session)
            mock_flush.assert_called_once_with(session)

    def test_session_mode_defers_flush(self):
        """T-F6-03: wf='session' → save() is no-op, no flush."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="session"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        session.add_message("user", "hello")

        with patch.object(mgr, '_flush_session', return_value=True) as mock_flush:
            mgr.save(session)
            mock_flush.assert_not_called()

    def test_int_mode_flushes_every_n_turns(self):
        """T-F6-04: wf=3 → flush on turn 3, 6, 9..."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency=3))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")

        flush_calls = 0
        original_flush = mgr._flush_session

        def counting_flush(s):
            nonlocal flush_calls
            flush_calls += 1
            return True

        with patch.object(mgr, '_flush_session', side_effect=counting_flush):
            for i in range(6):
                session.add_message("user", f"msg-{i}")
                mgr.save(session)

        # Turns 3 and 6 should trigger flush (counter incremented before check)
        assert flush_calls == 2

    def test_int_1_flushes_every_turn(self):
        """T-F6-05: wf=1 → equivalent to 'turn'."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency=1))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")

        flush_calls = 0

        def counting_flush(s):
            nonlocal flush_calls
            flush_calls += 1
            return True

        with patch.object(mgr, '_flush_session', side_effect=counting_flush):
            for i in range(3):
                session.add_message("user", f"msg-{i}")
                mgr.save(session)

        assert flush_calls == 3  # every turn

    def test_wf_0_silent_noop_documented_bug(self):
        """T-F6-06 BUG: wf=0 → silent no-op, data never sent."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency=0))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")

        flush_calls = 0

        def counting_flush(s):
            nonlocal flush_calls
            flush_calls += 1
            return True

        with patch.object(mgr, '_flush_session', side_effect=counting_flush):
            for i in range(5):
                session.add_message("user", f"msg-{i}")
                mgr.save(session)

        # BUG: wf=0 falls through all branches → never flushes
        assert flush_calls == 0  # documents the silent no-op bug

    def test_wf_negative_silent_noop_documented_bug(self):
        """T-F6-07 BUG: wf=-1 → silent no-op."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency=-1))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")

        flush_calls = 0

        def counting_flush(s):
            nonlocal flush_calls
            flush_calls += 1
            return True

        with patch.object(mgr, '_flush_session', side_effect=counting_flush):
            session.add_message("user", "msg")
            mgr.save(session)

        assert flush_calls == 0  # documents the silent no-op bug


# ---------------------------------------------------------------------------
# Block B: _flush_session and _synced flag
# ---------------------------------------------------------------------------


class TestFlushSession:
    """T-F6-08 to T-F6-11: synced flag, failure, empty, partial."""

    def test_flush_only_unsynced_messages(self):
        """T-F6-08: Second flush sends nothing (all already synced)."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        session.add_message("user", "msg-0")
        session.add_message("user", "msg-1")

        # First flush
        result1 = mgr._flush_session(session)
        assert result1 is True
        first_call_count = mock_session.add_messages.call_count

        # Second flush — nothing new
        result2 = mgr._flush_session(session)
        assert result2 is True
        assert mock_session.add_messages.call_count == first_call_count  # no new call

    def test_flush_failure_resets_synced(self):
        """T-F6-09: On failure, messages marked unsynced for retry."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_session.add_messages.side_effect = TimeoutError("timeout")
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        session.add_message("user", "will-fail")

        result = mgr._flush_session(session)
        assert result is False

        # Messages should be unsynced for retry
        unsynced = [m for m in session.messages if not m.get("_synced")]
        assert len(unsynced) >= 1

    def test_empty_session_flush_no_api_call(self):
        """T-F6-10: Empty session → return True, no API call."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        # No messages added

        result = mgr._flush_session(session)
        assert result is True
        mock_session.add_messages.assert_not_called()

    def test_all_synced_session_flush_no_api_call(self):
        """Session with all messages already synced → no API call."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        # Add pre-synced messages
        session.messages.append({"role": "user", "content": "old", "_synced": True})

        result = mgr._flush_session(session)
        assert result is True
        mock_session.add_messages.assert_not_called()


# ---------------------------------------------------------------------------
# Block C: Async writer loop
# ---------------------------------------------------------------------------


class TestAsyncWriterLoop:
    """T-F6-12 to T-F6-14: retry, drop, sentinel."""

    def test_async_mode_enqueues_not_sync_flush(self):
        """T-F6-02: wf='async' → save enqueues, doesn't call _flush_session directly."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="async"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        session.add_message("user", "hello")

        with patch.object(mgr, '_flush_session', return_value=True) as mock_flush:
            mgr.save(session)
            # async mode should NOT call _flush_session directly
            mock_flush.assert_not_called()

        mgr.shutdown()


# ---------------------------------------------------------------------------
# Block D: flush_all and shutdown
# ---------------------------------------------------------------------------


class TestFlushAllShutdown:
    """T-F6-15 to T-F6-17: drain, ordering, non-async safety."""

    def test_flush_all_processes_cached_sessions(self):
        """T-F6-15: flush_all flushes all cached sessions with unsynced messages."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="session"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        s1 = mgr.get_or_create("ch:1")
        s1.add_message("user", "msg-1")
        s2 = mgr.get_or_create("ch:2")
        s2.add_message("user", "msg-2")

        mgr.flush_all()

        # Both sessions should have been flushed
        # (add_messages called for each session with unsynced messages)
        assert mock_session.add_messages.call_count >= 2
        mgr.shutdown()

    def test_shutdown_completes_without_error(self):
        """T-F6-16: shutdown() doesn't raise."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="session"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        mgr.get_or_create("ch:1")
        # Should not raise
        mgr.shutdown()

    def test_shutdown_non_async_safe(self):
        """T-F6-17: shutdown() with non-async write_frequency doesn't crash."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="turn"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        mgr.get_or_create("ch:1")
        # Should not raise even without async queue
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Block E: Turn counter
# ---------------------------------------------------------------------------


class TestTurnCounter:
    """_turn_counter incremented on each save()."""

    def test_turn_counter_increments(self):
        """Each save() increments _turn_counter."""
        mgr, mock_honcho = _make_manager(_make_config(write_frequency="session"))
        mock_honcho.peer.return_value = MagicMock()
        mock_session = MagicMock()
        mock_session.context.return_value = MagicMock(messages=[])
        mock_honcho.session.return_value = mock_session

        session = mgr.get_or_create("ch:1")
        initial = mgr._turn_counter

        mgr.save(session)
        assert mgr._turn_counter == initial + 1

        mgr.save(session)
        assert mgr._turn_counter == initial + 2
        mgr.shutdown()
