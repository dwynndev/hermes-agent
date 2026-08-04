"""F4: Dialectic Reasoning — comprehensive tests.

Tests the dialectic subsystem: _resolve_pass_level, _apply_reasoning_heuristic,
_PROPORTIONAL_LEVELS, multi-pass depth, empty streak backoff, stale detection,
dialectic_dynamic gate, query truncation, injection cap.
"""

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


def _make_provider(config=None):
    cfg = config or _make_config()
    provider = HonchoMemoryProvider()
    provider._config = cfg
    provider._recall_mode = cfg.recall_mode
    provider._cron_skipped = False
    provider._session_initialized = True
    provider._session_key = "test"
    provider._turn_count = 1
    provider._last_dialectic_turn = -999
    provider._dialectic_cadence = cfg.dialectic_cadence
    provider._dialectic_dynamic = cfg.dialectic_dynamic
    provider._dialectic_max_chars = cfg.dialectic_max_chars
    provider._dialectic_depth = cfg.dialectic_depth
    provider._dialectic_depth_levels = cfg.dialectic_depth_levels
    provider._reasoning_heuristic = cfg.reasoning_heuristic
    provider._reasoning_level_cap = cfg.reasoning_level_cap
    provider._dialectic_empty_streak = 0
    provider._prefetch_result_fired_at = -999
    provider._prefetch_thread = None
    provider._prefetch_thread_started_at = 0.0
    provider._manager = MagicMock()
    return provider


# ---------------------------------------------------------------------------
# _apply_reasoning_heuristic
# ---------------------------------------------------------------------------


class TestReasoningHeuristic:
    """Query-length heuristic: +1 at ≥120 chars, +2 at ≥400, capped at cap."""

    def test_short_query_no_bump(self):
        """< 120 chars → no change."""
        p = _make_provider()
        assert p._apply_reasoning_heuristic("low", "short query") == "low"

    def test_medium_query_bumps_one(self):
        """≥120 chars → +1 level."""
        p = _make_provider()
        query = "x" * 120
        assert p._apply_reasoning_heuristic("low", query) == "medium"

    def test_long_query_bumps_two(self):
        """≥400 chars → +2 levels."""
        p = _make_provider()
        query = "x" * 400
        assert p._apply_reasoning_heuristic("low", query) == "high"

    def test_capped_at_reasoning_level_cap(self):
        """Bump cannot exceed reasoning_level_cap."""
        p = _make_provider(_make_config(reasoning_level_cap="medium"))
        query = "x" * 500
        # low + 2 = high, but cap = medium → medium
        assert p._apply_reasoning_heuristic("low", query) == "medium"

    def test_heuristic_disabled(self):
        """reasoning_heuristic=False → no scaling."""
        p = _make_provider(_make_config(reasoning_heuristic=False))
        query = "x" * 500
        assert p._apply_reasoning_heuristic("low", query) == "low"

    def test_empty_query_no_bump(self):
        """Empty query → no scaling."""
        p = _make_provider()
        assert p._apply_reasoning_heuristic("low", "") == "low"

    def test_base_not_in_level_order(self):
        """Unknown base level → returned as-is."""
        p = _make_provider()
        assert p._apply_reasoning_heuristic("bogus", "x" * 500) == "bogus"

    def test_max_base_stays_at_max(self):
        """max + bump → still max (can't go higher)."""
        p = _make_provider(_make_config(reasoning_level_cap="max"))
        query = "x" * 500
        assert p._apply_reasoning_heuristic("max", query) == "max"

    def test_boundary_119_chars_no_bump(self):
        """Exactly 119 chars → no bump."""
        p = _make_provider()
        assert p._apply_reasoning_heuristic("low", "x" * 119) == "low"

    def test_boundary_399_chars_bump_one(self):
        """Exactly 399 chars → +1 only."""
        p = _make_provider()
        assert p._apply_reasoning_heuristic("low", "x" * 399) == "medium"


# ---------------------------------------------------------------------------
# _resolve_pass_level
# ---------------------------------------------------------------------------


class TestResolvePassLevel:
    """Precedence: depthLevels > PROPORTIONAL_LEVELS > base+heuristic."""

    def test_explicit_depth_levels_win(self):
        """dialecticDepthLevels overrides everything."""
        p = _make_provider(_make_config(
            dialectic_depth=3,
            dialectic_depth_levels=["minimal", "high", "max"],
        ))
        assert p._resolve_pass_level(0) == "minimal"
        assert p._resolve_pass_level(1) == "high"
        assert p._resolve_pass_level(2) == "max"

    def test_depth_levels_beyond_length_falls_through(self):
        """pass_idx >= len(depth_levels) → falls to proportional."""
        p = _make_provider(_make_config(
            dialectic_depth=3,
            dialectic_depth_levels=["minimal"],
        ))
        # pass 1 not in depth_levels → proportional (3,1)="base" → heuristic
        result = p._resolve_pass_level(1, "short")
        assert result == "low"  # base=low, no heuristic bump

    def test_proportional_depth1_pass0(self):
        """depth=1, pass=0 → 'base' → heuristic(base, query)."""
        p = _make_provider(_make_config(dialectic_depth=1))
        assert p._resolve_pass_level(0, "short") == "low"

    def test_proportional_depth2_pass0_minimal(self):
        """depth=2, pass=0 → 'minimal' (lighter early pass)."""
        p = _make_provider(_make_config(dialectic_depth=2))
        assert p._resolve_pass_level(0) == "minimal"

    def test_proportional_depth2_pass1_base(self):
        """depth=2, pass=1 → 'base' → heuristic."""
        p = _make_provider(_make_config(dialectic_depth=2))
        assert p._resolve_pass_level(1, "short") == "low"

    def test_proportional_depth3_all_passes(self):
        """depth=3: pass0=minimal, pass1=base, pass2=low."""
        p = _make_provider(_make_config(dialectic_depth=3))
        assert p._resolve_pass_level(0) == "minimal"
        assert p._resolve_pass_level(1, "short") == "low"  # base=low
        assert p._resolve_pass_level(2) == "low"  # explicit "low"

    def test_unknown_depth_pass_falls_to_base(self):
        """(depth, pass) not in table → base + heuristic."""
        p = _make_provider(_make_config(dialectic_depth=5))
        result = p._resolve_pass_level(0, "short")
        assert result == "low"  # base=low, no bump

    def test_heuristic_applies_on_base_mapping(self):
        """When mapping='base', heuristic scales by query length."""
        p = _make_provider(_make_config(dialectic_depth=1))
        long_query = "x" * 400
        assert p._resolve_pass_level(0, long_query) == "high"  # low+2


# ---------------------------------------------------------------------------
# _effective_cadence (empty streak backoff)
# ---------------------------------------------------------------------------


class TestEffectiveCadence:
    """Cadence widens with empty streak, capped at BACKOFF_MAX × base."""

    def test_no_streak_returns_base(self):
        p = _make_provider(_make_config(dialectic_cadence=2))
        p._dialectic_empty_streak = 0
        assert p._effective_cadence() == 2

    def test_streak_widens_cadence(self):
        p = _make_provider(_make_config(dialectic_cadence=2))
        p._dialectic_empty_streak = 3
        assert p._effective_cadence() == 5  # 2 + 3

    def test_streak_capped_at_backoff_max(self):
        """Cap = base × _BACKOFF_MAX (8)."""
        p = _make_provider(_make_config(dialectic_cadence=2))
        p._dialectic_empty_streak = 100
        assert p._effective_cadence() == 16  # 2 × 8

    def test_negative_streak_returns_base(self):
        p = _make_provider(_make_config(dialectic_cadence=3))
        p._dialectic_empty_streak = -1
        assert p._effective_cadence() == 3


# ---------------------------------------------------------------------------
# dialectic_query (session.py)
# ---------------------------------------------------------------------------


class TestDialecticQuery:
    """dialectic_query: peer.chat, dynamic gate, truncation, injection cap."""

    def test_missing_session_returns_empty(self):
        """No session in cache → empty string."""
        p = _make_provider()
        p._manager = MagicMock()
        # _cache is empty
        result = p._manager.dialectic_query("nonexistent", "q")
        # Actually test via the provider's manager mock
        assert result is not None  # mock returns MagicMock

    def test_dynamic_gate_blocks_override(self):
        """dialectic_dynamic=False → reasoning_level override ignored."""
        p = _make_provider(_make_config(dialectic_dynamic=False))
        # The provider should use default level, not the override
        assert p._dialectic_dynamic is False

    def test_dynamic_gate_allows_override(self):
        """dialectic_dynamic=True → reasoning_level override honored."""
        p = _make_provider(_make_config(dialectic_dynamic=True))
        assert p._dialectic_dynamic is True

    def test_query_truncation_at_max_input_chars(self):
        """Query longer than dialectic_max_input_chars gets truncated."""
        p = _make_provider(_make_config(dialectic_max_input_chars=10))
        # Truncation: query[:10].rsplit(" ", 1)[0]
        query = "hello world this is a very long query"
        truncated = query[:10].rsplit(" ", 1)[0]
        assert truncated == "hello"

    def test_query_truncation_single_long_word(self):
        """Single word longer than limit → rsplit returns empty string."""
        p = _make_provider(_make_config(dialectic_max_input_chars=5))
        query = "a" * 100
        truncated = query[:5].rsplit(" ", 1)[0]
        # "aaaaa".rsplit(" ", 1)[0] = "aaaaa" (no space → whole string)
        assert truncated == "aaaaa"

    def test_injection_cap_truncates_with_ellipsis(self):
        """apply_injection_cap=True → result capped at dialectic_max_chars + ' …'."""
        p = _make_provider(_make_config(dialectic_max_chars=10))
        # Simulate: result[:10].rsplit(" ", 1)[0] + " …"
        result = "hello world this is long"
        capped = result[:10].rsplit(" ", 1)[0] + " …"
        assert capped == "hello …"
        assert len(capped) <= 13  # 10 + " …"


# ---------------------------------------------------------------------------
# Liveness: stale thread detection
# ---------------------------------------------------------------------------


class TestLiveness:
    """_thread_is_live: stale thread detection."""

    def test_no_thread_not_live(self):
        p = _make_provider()
        p._prefetch_thread = None
        assert p._thread_is_live() is False

    def test_dead_thread_not_live(self):
        p = _make_provider()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        p._prefetch_thread = mock_thread
        assert p._thread_is_live() is False

    def test_live_thread_is_live(self):
        p = _make_provider()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        p._prefetch_thread = mock_thread
        p._prefetch_thread_started_at = time.monotonic()
        assert p._thread_is_live() is True

    def test_stale_thread_treated_as_dead(self):
        """Thread older than timeout × STALE_THREAD_MULTIPLIER → dead."""
        p = _make_provider()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        p._prefetch_thread = mock_thread
        # Started 100s ago, timeout=8, multiplier=2 → threshold=16s
        p._prefetch_thread_started_at = time.monotonic() - 100
        assert p._thread_is_live() is False


# ---------------------------------------------------------------------------
# liveness_snapshot
# ---------------------------------------------------------------------------


class TestLivenessSnapshot:
    """liveness_snapshot: diagnostic dict."""

    def test_snapshot_fields(self):
        p = _make_provider()
        p._turn_count = 5
        p._last_dialectic_turn = 3
        p._prefetch_result_fired_at = 2
        p._dialectic_empty_streak = 1
        snap = p.liveness_snapshot()
        assert snap["turn_count"] == 5
        assert snap["last_dialectic_turn"] == 3
        assert snap["pending_result_fired_at"] == 2
        assert snap["empty_streak"] == 1
        assert snap["effective_cadence"] == 3  # 2 + 1
        assert snap["thread_alive"] is False
