"""Behavioral tests for TurnRetryState (T13-F1: turn-retry state machine).

Complements the shape-pinning tests in test_turn_retry_state.py. These pin
the semantics the turn loop actually relies on:

* every fresh instance starts with ALL guards False — a leaked True would
  silently skip a recovery branch forever (the loop checks-then-sets)
* instances are independent (no shared/aliased state across attempts)
* __iter__ yields exactly (name, value) tuples for every field — the
  debugging/dump contract
* the one-shot check-then-set pattern fires its recovery exactly once
"""

from __future__ import annotations

from dataclasses import fields

from agent.turn_retry_state import TurnRetryState

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
        # Would catch a mutable-class-default bug (shared dict/list) if a
        # future field type changed — and a __post_init__ that aliases state.
        a = TurnRetryState()
        b = TurnRetryState()
        a.has_retried_429 = True
        a.restart_with_compressed_messages = True
        assert b.has_retried_429 is False
        assert b.restart_with_compressed_messages is False


class TestIterationProtocol:
    def test_iter_yields_name_value_tuples_for_every_field(self):
        # Pins the (name, value) TUPLE contract: dict() over the iterator
        # works only if each item is a 2-element sequence. A __iter__ that
        # yielded bare names, or (value, name) pairs, fails here.
        s = TurnRetryState()
        s.copilot_stale_cred_retry_attempted = True
        pairs = list(s)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        assert dict(pairs) == dict(s)
        expected_names = {f.name for f in fields(TurnRetryState)}
        assert {name for name, _ in pairs} == expected_names
        assert dict(pairs)["copilot_stale_cred_retry_attempted"] is True

    def test_iter_reflects_live_mutation(self):
        s = TurnRetryState()
        before = dict(s)["restart_with_length_continuation"]
        s.restart_with_length_continuation = True
        after = dict(s)["restart_with_length_continuation"]
        assert before is False and after is True


class TestOneShotGuardPattern:
    def test_check_then_set_fires_exactly_once(self):
        # The turn loop's actual usage pattern for every guard:
        #     if not state.X:
        #         state.X = True
        #         <recovery branch>
        # The recovery must execute exactly once per attempt regardless of
        # how many times the branch condition is re-encountered. A guard
        # that reset itself (or a field read that didn't persist the write)
        # would fire the recovery repeatedly — this catches that.
        s = TurnRetryState()
        fires = 0
        for _ in range(3):  # loop re-encounters the branch 3 times
            if not s.thinking_sig_retry_attempted:
                s.thinking_sig_retry_attempted = True
                fires += 1
        assert fires == 1
        assert s.thinking_sig_retry_attempted is True
        # Other guards in the same attempt are untouched — the loop may
        # still fire them later within this attempt.
        assert s.has_retried_429 is False
        assert s.restart_with_compressed_messages is False
