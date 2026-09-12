"""W55 B06 — same-correlation concurrent launches must be deduplicated (W54-F002).

Pre-fix, the duplicate-correlation check and the registry write are two separate
lock sections with the child build between them, so two concurrent launches that
share a correlation_id both pass the check and both register. This test forces
both threads into the build span via a barrier-gated build function and asserts
exactly one launch claims the correlation key and the registry holds one entry.
"""  # noqa: D205,D400,D415

import threading
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import (
    _REGISTRY,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
)


class _FakeChild:
    def __init__(self, ident):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.interrupted = False

    def interrupt(self, _reason):
        self.interrupted = True

    def hard_interrupt(self, reason, *, tool_reason=None):
        self.interrupted = True


@pytest.fixture
def service(monkeypatch):
    parent = SimpleNamespace(session_id="parent-b06", enabled_toolsets=["file"])
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "stub",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    return SubagentLifecycleService(lambda: parent)


def test_same_correlation_concurrent_launch_exactly_one_wins(service, monkeypatch):
    barrier = threading.Barrier(2)
    ids = iter(["sa-b06-a", "sa-b06-b"])

    def gated_build(**_kwargs):
        # Both threads rendezvous here. Pre-fix both are provably inside the
        # unlocked build span before either one writes the correlations dict;
        # post-fix only the winner ever reaches the build.
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return _FakeChild(next(ids))

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", gated_build)

    outcomes = []

    def attempt():
        try:
            handle = service.launch(SubagentLaunchRequest(goal="race-b06", correlation_id="corr-b06"))
            outcomes.append(("ok", handle))
        except SubagentLifecycleError as exc:
            outcomes.append(("error", str(exc)))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    oks = [outcome for outcome in outcomes if outcome[0] == "ok"]
    errors = [outcome for outcome in outcomes if outcome[0] == "error"]

    assert len(oks) == 1, outcomes
    assert len(errors) == 1, outcomes
    assert errors[0][1] == "Duplicate correlation_id for this parent session."

    winner = oks[0][1]
    key = ("parent-b06", "corr-b06")
    # The registry holds exactly one entry and it belongs to the winner; the
    # correlation mapping points at the winner's real subagent_id (never a
    # placeholder).
    assert set(_REGISTRY.records) == {winner.subagent_id}
    claimed = _REGISTRY.correlations.get(key)
    assert isinstance(claimed, str), claimed
    assert claimed == winner.subagent_id

    # Build failure must release the claim so a retry with the same
    # correlation_id is not poisoned (rollback contract of the fix).

    def failing_build(**_kwargs):
        raise RuntimeError("child build exploded")

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", failing_build)
    with pytest.raises(RuntimeError, match="child build exploded"):
        service.launch(SubagentLaunchRequest(goal="boom-b06", correlation_id="corr-retry"))

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: _FakeChild("sa-b06-retry"),
    )
    retried = service.launch(SubagentLaunchRequest(goal="retry-b06", correlation_id="corr-retry"))
    assert _REGISTRY.correlations[("parent-b06", "corr-retry")] == retried.subagent_id