"""Mid-turn model switches are deferred, not refused.

`agent.switch_model()` mutates model/provider/base_url/client in place while
the worker thread reads them, so a live swap can pair a new base_url with an
old model. That's a reason to DELAY the swap — the previous behavior returned
error 4009 ("session busy — /interrupt the current turn before switching
models"), which made the picker useless exactly when you most want it: watching
a turn go wrong and reaching for a different model.

config.set now parks the pick and reports scope="pending" so the UI paints it
immediately; the turn-completion path applies it before the next prompt runs.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import tui_gateway.server as server


def _session(running: bool) -> dict:
    return {
        "agent": object(),
        "history": [],
        "history_lock": threading.RLock(),
        "running": running,
        "session_key": "sid-1",
    }


def _config_set(sid: str) -> dict:
    return server._methods["config.set"](
        "rid-1", {"session_id": sid, "key": "model", "value": "grok-4.5 --provider xai --session"}
    )


class TestMidTurnModelSwitchDefers:
    def test_busy_session_parks_the_pick_instead_of_erroring(self) -> None:
        session = _session(running=True)

        with patch.dict(server._sessions, {"sid-1": session}, clear=False):
            resp = _config_set("sid-1")

        assert "error" not in resp, f"a busy session must not reject the switch: {resp}"
        assert resp["result"]["scope"] == "pending"
        assert resp["result"]["value"] == "grok-4.5 --provider xai --session"
        assert session["pending_model_switch"]["value"] == "grok-4.5 --provider xai --session"

    def test_idle_session_still_applies_immediately(self) -> None:
        session = _session(running=False)

        with (
            patch.dict(server._sessions, {"sid-1": session}, clear=False),
            patch.object(
                server,
                "_apply_model_switch",
                return_value={"value": "grok-4.5", "warning": "", "scope": "session"},
            ) as apply_now,
        ):
            resp = _config_set("sid-1")

        assert apply_now.called
        assert resp["result"]["scope"] == "session"
        assert "pending_model_switch" not in session


class TestPendingSwitchAppliesWhenTheTurnEnds:
    def test_applied_once_and_republished(self) -> None:
        session = _session(running=False)
        session["pending_model_switch"] = {"value": "grok-4.5 --provider xai --session", "confirm_expensive_model": False}

        with (
            patch.object(server, "_apply_model_switch", return_value={"value": "grok-4.5", "warning": ""}) as applied,
            patch.object(server, "_session_info", return_value={"model": "grok-4.5"}),
            patch.object(server, "_emit") as emit,
        ):
            server._apply_pending_model_switch("sid-1", session)
            # A second drain must not re-apply — the pick is claimed, not left
            # to fire again on every subsequent turn.
            server._apply_pending_model_switch("sid-1", session)

        assert applied.call_count == 1
        assert applied.call_args.args[2] == "grok-4.5 --provider xai --session"
        assert "pending_model_switch" not in session
        assert [call.args[0] for call in emit.call_args_list] == ["session.info"]

    def test_still_running_leaves_the_pick_parked(self) -> None:
        session = _session(running=True)
        session["pending_model_switch"] = {"value": "grok-4.5 --provider xai --session"}

        with patch.object(server, "_apply_model_switch") as applied:
            server._apply_pending_model_switch("sid-1", session)

        assert not applied.called
        assert session["pending_model_switch"]

    def test_a_failed_switch_is_swallowed_so_the_next_turn_still_runs(self) -> None:
        session = _session(running=False)
        session["pending_model_switch"] = {"value": "bogus-model --provider nope --session"}

        with (
            patch.object(server, "_apply_model_switch", side_effect=RuntimeError("no such model")),
            patch.object(server, "_session_info", return_value={}),
            patch.object(server, "_emit"),
        ):
            server._apply_pending_model_switch("sid-1", session)

        assert "pending_model_switch" not in session

    def test_no_pending_switch_is_a_no_op(self) -> None:
        session = _session(running=False)

        with patch.object(server, "_apply_model_switch") as applied, patch.object(server, "_emit") as emit:
            server._apply_pending_model_switch("sid-1", session)

        assert not applied.called
        assert not emit.called
