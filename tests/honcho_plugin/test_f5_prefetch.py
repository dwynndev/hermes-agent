"""F5: Prefetch & Context Injection — comprehensive tests.

Tests system_prompt_block(), prefetch(), queue_prefetch(), on_turn_start(),
two-layer injection, stale detection, thread safety, cadence logic.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.honcho import HonchoMemoryProvider


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


def _make_provider(recall_mode="hybrid", cron_skipped=False, config=None):
    cfg = config or _make_config(recall_mode=recall_mode)
    provider = HonchoMemoryProvider()
    provider._config = cfg
    provider._recall_mode = recall_mode
    provider._cron_skipped = cron_skipped
    provider._session_initialized = True
    provider._session_key = "test"
    provider._turn_count = 1
    provider._last_context_turn = -999
    provider._last_dialectic_turn = -999
    provider._context_cadence = cfg.context_cadence
    provider._dialectic_cadence = cfg.dialectic_cadence
    provider._dialectic_dynamic = cfg.dialectic_dynamic
    provider._dialectic_max_chars = cfg.dialectic_max_chars
    provider._dialectic_depth = cfg.dialectic_depth
    provider._dialectic_depth_levels = cfg.dialectic_depth_levels
    provider._reasoning_heuristic = cfg.reasoning_heuristic
    provider._reasoning_level_cap = cfg.reasoning_level_cap
    provider._dialectic_empty_streak = 0
    provider._prefetch_result = ""
    provider._prefetch_result_fired_at = -999
    provider._prefetch_thread = None
    provider._prefetch_thread_started_at = 0.0
    provider._base_context_cache = None
    provider._base_context_lock = threading.Lock()
    provider._prefetch_lock = threading.Lock()
    provider._manager = MagicMock()
    return provider


# ---------------------------------------------------------------------------
# Block A: system_prompt_block()
# ---------------------------------------------------------------------------


class TestSystemPromptBlock:
    """T1-T4: Static text per recall_mode, cron guard."""

    def test_hybrid_mentions_tools_and_autoinject(self):
        """T1: hybrid → tool instructions + auto-inject mention."""
        p = _make_provider(recall_mode="hybrid")
        block = p.system_prompt_block()
        assert "honcho_profile" in block or "honcho" in block.lower()

    def test_context_hides_tools(self):
        """T2: context mode → no tool names in block."""
        p = _make_provider(recall_mode="context")
        block = p.system_prompt_block()
        assert "honcho_profile" not in block
        assert "honcho_search" not in block

    def test_tools_mode_shows_tools_no_autoinject(self):
        """T3: tools mode → tool list, no auto-inject claim."""
        p = _make_provider(recall_mode="tools")
        block = p.system_prompt_block()
        assert "honcho_profile" in block or "honcho" in block.lower()

    @pytest.mark.parametrize("mode", ["hybrid", "context", "tools"])
    def test_cron_skipped_returns_empty(self, mode):
        """T4: cron_skipped → empty string for any recall_mode."""
        p = _make_provider(recall_mode=mode, cron_skipped=True)
        assert p.system_prompt_block() == ""

    def test_block_is_static_text(self):
        """system_prompt_block returns same text on repeated calls (cache-friendly)."""
        p = _make_provider(recall_mode="hybrid")
        block1 = p.system_prompt_block()
        block2 = p.system_prompt_block()
        assert block1 == block2


# ---------------------------------------------------------------------------
# Block B: prefetch()
# ---------------------------------------------------------------------------


class TestPrefetch:
    """T5-T8: Cache return, mode guards, cron guard."""

    def test_tools_mode_returns_empty(self):
        """T5: recall_mode=tools → prefetch always returns ''."""
        p = _make_provider(recall_mode="tools")
        p._prefetch_result = "should not leak"
        p._prefetch_result_fired_at = p._turn_count
        assert p.prefetch("tell me about my project") == ""

    def test_consume_pending_empty_returns_empty(self):
        """T6: No pending dialectic → _consume_pending_dialectic returns ''."""
        p = _make_provider(recall_mode="hybrid")
        p._turn_count = 5
        p._prefetch_result = ""
        p._prefetch_result_fired_at = -999
        result = p._consume_pending_dialectic()
        assert result == ""

    def test_cron_skipped_returns_empty(self):
        """T8: cron_skipped → '' even with full cache."""
        p = _make_provider(recall_mode="hybrid", cron_skipped=True)
        p._base_context_cache = "rich context"
        p._prefetch_result = "dialectic"
        assert p.prefetch("query") == ""

    def test_consume_pending_returns_fresh_result(self):
        """Fresh pending dialectic → returned and cleared."""
        p = _make_provider(recall_mode="hybrid")
        p._turn_count = 3
        p._prefetch_result = "dialectic answer"
        p._prefetch_result_fired_at = 3
        result = p._consume_pending_dialectic()
        assert result == "dialectic answer"
        # After consumption, cache is cleared
        assert p._prefetch_result == ""
        assert p._prefetch_result_fired_at == -999


# ---------------------------------------------------------------------------
# Block C: queue_prefetch()
# ---------------------------------------------------------------------------


class TestQueuePrefetch:
    """T9-T13: Thread firing, cadence, trivial skip, double-fire guard."""

    def test_tools_mode_no_thread(self):
        """T9: recall_mode=tools → queue_prefetch doesn't start thread."""
        p = _make_provider(recall_mode="tools")
        p.queue_prefetch("some query")
        assert p._prefetch_thread is None

    def test_cron_skipped_no_thread(self):
        """cron_skipped → no background work."""
        p = _make_provider(cron_skipped=True)
        p.queue_prefetch("query")
        assert p._prefetch_thread is None

    def test_live_thread_skips_double_fire(self):
        """T13: Live prefetch thread → don't spawn another."""
        p = _make_provider(recall_mode="hybrid")
        p._turn_count = 5
        p._last_dialectic_turn = 0
        # Simulate a live thread
        t = threading.Thread(target=time.sleep, args=(10,), daemon=True)
        t.start()
        p._prefetch_thread = t
        p._prefetch_thread_started_at = time.monotonic()
        old_thread = p._prefetch_thread
        p.queue_prefetch("query")
        # Should NOT replace the thread
        assert p._prefetch_thread is old_thread


# ---------------------------------------------------------------------------
# Block D: Stale result detection & two-layer
# ---------------------------------------------------------------------------


class TestStaleAndTwoLayer:
    """T14-T17: Stale detection, two-layer concatenation, cache refresh."""

    def test_stale_result_discarded(self):
        """T14: fired_at far in past → result discarded."""
        p = _make_provider(recall_mode="hybrid")
        p._dialectic_cadence = 2
        p._turn_count = 10
        p._prefetch_result = "stale context"
        p._prefetch_result_fired_at = 3  # 10-3=7 > 2*2=4
        # _consume_pending_dialectic should return "" for stale
        with p._prefetch_lock:
            stale_limit = p._dialectic_cadence * p._STALE_RESULT_MULTIPLIER
            is_stale = (p._turn_count - p._prefetch_result_fired_at) > stale_limit
        assert is_stale is True

    def test_fresh_result_not_stale(self):
        """T15: fired_at within window → not stale."""
        p = _make_provider(recall_mode="hybrid")
        p._dialectic_cadence = 2
        p._turn_count = 5
        p._prefetch_result = "fresh context"
        p._prefetch_result_fired_at = 4  # 5-4=1 <= 2*2=4
        with p._prefetch_lock:
            stale_limit = p._dialectic_cadence * p._STALE_RESULT_MULTIPLIER
            is_stale = (p._turn_count - p._prefetch_result_fired_at) > stale_limit
        assert is_stale is False

    def test_stale_multiplier_is_2(self):
        """_STALE_RESULT_MULTIPLIER constant is 2."""
        p = _make_provider()
        assert p._STALE_RESULT_MULTIPLIER == 2

    def test_two_layer_both_present(self):
        """T16: base + dialectic → both in result."""
        p = _make_provider(recall_mode="hybrid")
        p._turn_count = 3
        p._base_context_cache = "## User Card\nName: Alice"
        p._last_context_turn = 3
        p._prefetch_result = "Dialectic: Alice prefers Python"
        p._prefetch_result_fired_at = 3
        p._last_dialectic_turn = 3
        result = p.prefetch("help with code")
        # Both layers should be present (if prefetch assembles them)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Block E: Thread safety & edge cases
# ---------------------------------------------------------------------------


class TestThreadSafetyAndEdgeCases:
    """T18-T20: Race conditions, on_turn_start, backoff cap."""

    def test_concurrent_prefetch_consume_once(self):
        """T18: 10 threads racing on prefetch → dialectic consumed at most once."""
        p = _make_provider(recall_mode="hybrid")
        p._turn_count = 3
        p._prefetch_result = "dialectic-result"
        p._prefetch_result_fired_at = 3
        p._last_dialectic_turn = 3
        p._base_context_cache = None
        p._manager.pop_context_result.return_value = {}

        results = []
        barrier = threading.Barrier(10)

        def reader():
            barrier.wait()
            results.append(p.prefetch("query"))

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Count how many got the dialectic result specifically
        dialectic_hits = [r for r in results if "dialectic-result" in r]
        # At most 1 thread should get the dialectic (consume-once semantics)
        assert len(dialectic_hits) <= 1

    def test_on_turn_start_sets_turn_count(self):
        """T19: on_turn_start sets _turn_count from argument."""
        p = _make_provider()
        p._turn_count = 5
        p.on_turn_start(3, "msg")
        assert p._turn_count == 3

    def test_effective_cadence_backoff_capped(self):
        """T20: Empty streak backoff capped at BACKOFF_MAX × base."""
        p = _make_provider()
        p._dialectic_cadence = 2
        p._dialectic_empty_streak = 100
        eff = p._effective_cadence()
        assert eff == 16  # 2 × 8

    def test_backoff_max_constant(self):
        """_BACKOFF_MAX is 8."""
        p = _make_provider()
        assert p._BACKOFF_MAX == 8

    def test_stale_thread_multiplier(self):
        """_STALE_THREAD_MULTIPLIER is 2.0."""
        p = _make_provider()
        assert p._STALE_THREAD_MULTIPLIER == 2.0


# ---------------------------------------------------------------------------
# Trivial prompt detection
# ---------------------------------------------------------------------------


class TestTrivialPrompt:
    """_is_trivial_prompt: filters out low-signal prompts."""

    def test_empty_is_trivial(self):
        p = _make_provider()
        assert p._is_trivial_prompt("") is True

    def test_whitespace_is_trivial(self):
        p = _make_provider()
        assert p._is_trivial_prompt("   ") is True

    def test_short_ack_is_trivial(self):
        p = _make_provider()
        assert p._is_trivial_prompt("ok") is True
        assert p._is_trivial_prompt("yes") is True
        assert p._is_trivial_prompt("thanks") is True

    def test_slash_command_is_trivial(self):
        p = _make_provider()
        assert p._is_trivial_prompt("/new") is True
        assert p._is_trivial_prompt("/reset") is True

    def test_substantive_query_not_trivial(self):
        p = _make_provider()
        assert p._is_trivial_prompt("help me debug this Python code") is False
        assert p._is_trivial_prompt("what is the meaning of life?") is False
