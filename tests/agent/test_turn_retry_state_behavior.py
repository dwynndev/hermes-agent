"""Behavioral tests for TurnRetryState (T13-F1: turn-retry state machine).

Complements the shape-pinning tests in test_turn_retry_state.py. These pin
the semantics the turn loop actually relies on:

* every fresh instance starts with ALL guards False — a leaked True would
  silently skip a recovery branch forever (the loop checks-then-sets)
* instances are independent (dataclass default values are not shared state)
* __iter__ yields exactly (name, value) pairs for every field — the
  debugging/dump contract, and the basis for any future "reset all guards"
  tooling
* restart signals and auth guards are orthogonal flag groups (no coupling)
"""

from __future__ import annotations

from dataclasses import fields

from agent.turn_retry_state import TurnRetryState

GUARD_FIELDS = {
    "codex_auth_retry_attempted",
    "anthropic_auth_retry_attempted",
    "nous_auth_retry_attempted",
    "nous_paid_entitlement_refresh_attempted",
    "copilot_auth_retry_attempted",
    "copilot_stale_cred_retry_attempted",
    "vertex_auth_retry_attempted",
    "thinking_sig_retry_attempted",
    "invalid_encrypted_content_retry_attempted",
    "image_shrink_retry_attempted",
    "multimodal_tool_content_retry_attempted",
    "oauth_1m_beta_retry_attempted",
    "llama_cpp_grammar_retry_attempted",
    "primary_recovery_attempted",
    "has_retried_429",
    "auth_failover_attempted",
}
RESTART_FIELDS = {
    "restart_with_compressed_messages",
    "restart_with_length_continuation",
    "restart_with_rebuilt_messages",
    "restart_with_redirected_messages",
}


class TestFreshInstanceDefaults:
    def test_all_flags_start_false(self):
        # The loop's one-shot guarantee depends on this: a default-True guard
        # would make its recovery branch permanently unreachable.
        s = TurnRetryState()
        assert all(getattr(s, f.name) is False for f in fields(s))

    def test_no_restart_signal_raised_on_fresh_instance(self):
        s = TurnRetryState()
        for name in RESTART_FIELDS:
            assert getattr(s, name) is False, name


class TestInstanceIndependence:
    def test_mutation_does_not_leak_across_instances(self):
        a = TurnRetryState()
        b = TurnRetryState()
        a.has_retried_429 = True
        a.restart_with_compressed_messages = True
        assert b.has_retried_429 is False
        assert b.restart_with_compressed_messages is False

    def test_each_turn_gets_a_clean_slate(self):
        # Simulates the loop creating a fresh state per api_call_count.
        for _ in range(3):
            s = TurnRetryState()
            s.auth_failover_attempted = True
            s.restart_with_redirected_messages = True
        s2 = TurnRetryState()
        assert s2.auth_failover_attempted is False
        assert s2.restart_with_redirected_messages is False


class TestIterationProtocol:
    def test_iter_yields_name_value_pairs_for_every_field(self):
        s = TurnRetryState()
        s.copilot_stale_cred_retry_attempted = True
        pairs = dict(s)
        expected_names = {f.name for f in fields(TurnRetryState)}
        assert set(pairs) == expected_names
        assert pairs["copilot_stale_cred_retry_attempted"] is True
        assert pairs["has_retried_429"] is False

    def test_iter_reflects_live_mutation(self):
        s = TurnRetryState()
        before = dict(s)["restart_with_length_continuation"]
        s.restart_with_length_continuation = True
        after = dict(s)["restart_with_length_continuation"]
        assert before is False and after is True


class TestFlagOrthogonality:
    def test_auth_guard_and_restart_signal_do_not_interfere(self):
        # The loop sets auth guards on escalation and restart signals on
        # rebuild; they must be independently writable/readable.
        s = TurnRetryState()
        for name in sorted(GUARD_FIELDS):
            setattr(s, name, True)
        assert all(getattr(s, n) is False for n in RESTART_FIELDS)
        for name in RESTART_FIELDS:
            setattr(s, name, True)
        assert all(getattr(s, n) is True for n in GUARD_FIELDS)
