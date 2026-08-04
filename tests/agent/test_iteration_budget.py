"""Behavioral tests for IterationBudget (T13-F1: turn-retry state machine).

The budget gates every agent loop iteration — a consume/refund bug either
lets a runaway agent burn unlimited API calls (over-consume) or starves a
legitimate one (under-refund). These tests pin the actual invariants, not
just the API shape:

* consume() succeeds exactly max_total times, then refuses (hard boundary)
* refund() re-enables consumption after exhaustion (execute_code refund path)
* refund() below zero is clamped — never creates credit
* remaining never goes negative
* concurrent consume() never over-allocates past max_total (thread safety —
  the whole point of the lock; 8 threads race 500 consumes on a cap of 50)
"""

from __future__ import annotations

import threading

from agent.iteration_budget import IterationBudget


class TestConsumeBoundary:
    def test_consume_allowed_exactly_max_times(self):
        b = IterationBudget(max_total=3)
        results = [b.consume() for _ in range(5)]
        assert results == [True, True, True, False, False]
        assert b.used == 3
        assert b.remaining == 0

    def test_zero_budget_refuses_immediately(self):
        b = IterationBudget(max_total=0)
        assert b.consume() is False
        assert b.used == 0
        assert b.remaining == 0


class TestRefund:
    def test_refund_after_exhaustion_reenables_consume(self):
        b = IterationBudget(max_total=1)
        assert b.consume() is True
        assert b.consume() is False
        b.refund()
        assert b.used == 0
        assert b.consume() is True  # refunded iteration is spendable again

    def test_refund_below_zero_is_clamped(self):
        b = IterationBudget(max_total=2)
        b.refund()  # nothing was consumed — must NOT create credit
        b.refund()
        assert b.used == 0
        assert b.remaining == 2
        # still exactly max_total consumes allowed, not more
        assert [b.consume() for _ in range(3)] == [True, True, False]


class TestThreadSafety:
    def test_concurrent_consume_never_over_allocates(self):
        cap = 50
        b = IterationBudget(max_total=cap)
        grants: list[bool] = []
        grants_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            local = [b.consume() for _ in range(60)]
            with grants_lock:
                grants.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 8 threads x 60 attempts = 480 requests; exactly cap may succeed.
        assert sum(grants) == cap
        assert b.used == cap
        assert b.remaining == 0

    def test_concurrent_consume_refund_stays_consistent(self):
        b = IterationBudget(max_total=100)
        errors: list[str] = []

        def consumer():
            for _ in range(200):
                if b.consume() and b.remaining < 0:
                    errors.append("remaining went negative")

        def refunder():
            for _ in range(100):
                b.refund()

        ts = [threading.Thread(target=consumer) for _ in range(4)]
        ts.append(threading.Thread(target=refunder))
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        assert not errors
        assert 0 <= b.used <= 100
        assert b.remaining == max(0, 100 - b.used)
