"""Wave 12 F2 — per-task model override + per-model reasoning at spawn.

Contract tests for the two-model division of labor (AS-0018):
deepseek-v4-flash-0731 runs worker swarms, qwen3.8-max pins reviews.
delegate_task accepts an optional validated `model` param (top-level and
per-task) that overrides delegation.model for that child, and child
reasoning effort resolves per EFFECTIVE child model through
agent.reasoning_overrides when delegation.reasoning_effort is unset.

Hermetic: no API calls, no real agent construction (integration test
monkeypatches the child builder). Runs under the clone conftest's isolated
HERMES_HOME.
"""
from __future__ import annotations

import json
import types

import pytest

from tools import delegate_tool as dt


FLASH = "deepseek-v4-flash-0731"
QWEN = "qwen3.8-max"


def _err_msg(result: str) -> str:
    parsed = json.loads(result)
    assert "error" in parsed, f"expected a tool_error JSON, got: {parsed}"
    return parsed["error"]


# ---------------------------------------------------------------------------
# 1. Allowlist
# ---------------------------------------------------------------------------


PRO = "deepseek-v4-pro-0813"
FLASH2 = "qwen3.8-flash"


class TestModelAllowlist:
    def test_allowlist_is_exactly_the_quad_model_policy(self):
        # AS-0025 (Wave 40): qwen3.8-flash is the 4th allowed worker.
        assert dt.ALLOWED_WORKER_MODELS == frozenset({FLASH, QWEN, PRO, FLASH2})

    @pytest.mark.parametrize("model", [FLASH, QWEN, PRO, FLASH2])
    def test_allowed_models_pass(self, model):
        assert dt._validate_worker_model(model) is None

    def test_disallowed_model_rejected_with_actionable_message(self):
        err = dt._validate_worker_model("gpt-5")
        assert err is not None
        assert "gpt-5" in err
        assert FLASH in err and QWEN in err

    @pytest.mark.parametrize("bad", ["", 123, ["a"], {"model": "x"}])
    def test_non_string_or_empty_rejected(self, bad):
        assert dt._validate_worker_model(bad) is not None

    def test_none_means_no_override(self):
        # None = absent pin → valid (falls back to delegation.model).
        assert dt._validate_worker_model(None) is None


# ---------------------------------------------------------------------------
# 2. Batch validation covers task-level model fields
# ---------------------------------------------------------------------------


def _batch(goal_a="first long enough goal text", goal_b="second long enough goal text"):
    return [
        {"goal": goal_a, "model": FLASH},
        {"goal": goal_b},
    ]


class TestBatchModelValidation:
    def test_valid_batch_models_pass(self):
        assert dt._validate_batch_tasks(_batch()) is None

    def test_invalid_task_model_rejected_with_index(self):
        tasks = _batch()
        tasks[0]["model"] = "gpt-5"
        err = dt._validate_batch_tasks(tasks)
        assert err is not None
        assert "0" in err and "gpt-5" in err

    def test_single_task_form_not_batch_validated(self):
        # A one-entry array is the canonical single-task shape: upstream
        # dropped the "at least 2 tasks" minimum (post-merge audit of
        # #81141), so a single valid goal validates to None.
        assert dt._validate_batch_tasks([{"goal": "x" * 40}]) is None


# ---------------------------------------------------------------------------
# 3. Schema surface: model param visible to the LLM
# ---------------------------------------------------------------------------


class TestSchemaModelSurface:
    def test_top_level_model_property(self):
        props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model" in props
        assert props["model"]["type"] == "string"

    def test_task_item_model_property(self):
        items = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        assert "model" in items["properties"]

    def test_dynamic_overrides_document_model(self):
        overrides = dt._build_dynamic_schema_overrides()
        assert "model" in overrides["parameters"]["properties"]
        assert "model" in overrides["description"]

    def test_model_not_hidden_from_model(self):
        # 'model' must NOT be stripped like acp_command/acp_args are.
        assert "model" not in dt._MODEL_HIDDEN_TASK_FIELDS


# ---------------------------------------------------------------------------
# 4. Credential resolution honors per-task override
# ---------------------------------------------------------------------------


def _parent():
    p = types.SimpleNamespace()
    p.model = QWEN
    p.provider = "alibaba"
    p.base_url = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    return p


class TestCredentialOverride:
    def test_per_task_model_overrides_configured_model(self):
        creds = dt._resolve_delegation_credentials(
            {"model": FLASH}, _parent(), model_override=QWEN
        )
        assert creds["model"] == QWEN

    def test_no_override_keeps_configured_model(self):
        creds = dt._resolve_delegation_credentials({"model": FLASH}, _parent())
        assert creds["model"] == FLASH

    def test_override_with_base_url_branch_keeps_delegation_base_url(self):
        creds = dt._resolve_delegation_credentials(
            {
                "model": FLASH,
                "base_url": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            },
            _parent(),
            model_override=QWEN,
        )
        assert creds["model"] == QWEN
        assert "token-plan" in creds["base_url"]

    def test_override_with_empty_config_model(self):
        creds = dt._resolve_delegation_credentials({}, _parent(), model_override=FLASH)
        assert creds["model"] == FLASH


# ---------------------------------------------------------------------------
# 5. Per-model reasoning at spawn
# ---------------------------------------------------------------------------


class TestResolveChildReasoning:
    OVERRIDES = {FLASH: "low", QWEN: "max"}

    def _cfg(self, effort=None):
        agent = {"reasoning_overrides": dict(self.OVERRIDES), "reasoning_effort": "medium"}
        if effort is not None:
            agent["reasoning_effort"] = effort
        return {"agent": agent}

    def test_explicit_delegation_effort_wins_backward_compat(self):
        rc = dt._resolve_child_reasoning(
            {"reasoning_effort": "xhigh"}, self._cfg(), FLASH, {"effort": "max"}
        )
        assert rc is not None and rc.get("effort") == "xhigh"

    def test_flash_false_semantics_preserved(self):
        rc = dt._resolve_child_reasoning(
            {"reasoning_effort": False}, self._cfg(), FLASH, {"effort": "max"}
        )
        assert rc is not None and rc.get("enabled") is False

    def test_unset_effort_flash_child_gets_low_override(self):
        rc = dt._resolve_child_reasoning({}, self._cfg(), FLASH, None)
        assert rc is not None and rc.get("effort") == "low"

    def test_unset_effort_qwen_child_gets_max_override(self):
        rc = dt._resolve_child_reasoning({}, self._cfg(), QWEN, None)
        assert rc is not None and rc.get("effort") == "max"

    def test_unset_effort_no_override_falls_to_global(self):
        rc = dt._resolve_child_reasoning({}, self._cfg(), "some-other-model", None)
        assert rc is not None and rc.get("effort") == "medium"

    def test_unset_effort_no_config_falls_to_parent(self):
        parent_rc = {"enabled": True, "effort": "high"}
        rc = dt._resolve_child_reasoning({}, {}, FLASH, parent_rc)
        assert rc is parent_rc


# ---------------------------------------------------------------------------
# 6. Integration: delegate_task threads per-task models to the child builder
# ---------------------------------------------------------------------------


class TestDelegateTaskModelThreading:
    @pytest.fixture()
    def harness(self, monkeypatch):
        """Patch the heavy machinery; capture child-builder kwargs."""
        built = []

        def fake_build(**kwargs):
            built.append(kwargs)
            child = types.SimpleNamespace()
            child.session_id = f"fake-{len(built)}"
            child.model = kwargs.get("model")
            child.tool_progress_callback = None
            return child

        run_calls = []

        def fake_run(task_index, goal, child, parent_agent, **kw):
            run_calls.append({"task_index": task_index, "model": getattr(child, "model", None)})
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": "ok",
                "model": getattr(child, "model", None),
                "api_calls": 1,
                "duration_seconds": 0.1,
            }

        monkeypatch.setattr(dt, "is_spawn_paused", lambda: False)
        monkeypatch.setattr(dt, "_load_config", lambda: {"model": FLASH, "max_iterations": 5})
        monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_build)
        monkeypatch.setattr(dt, "_run_single_child", fake_run)
        # Function-local imports — patch the source modules.
        import tools.delegation_live_log as dll

        monkeypatch.setattr(dll, "create_live_transcripts", lambda tl, ctx, **k: (None, [], []))
        monkeypatch.setattr(dll, "update_manifest_statuses", lambda *a, **k: None)
        parent = _parent()
        parent._delegate_depth = 0
        parent._fallback_chain = [QWEN, FLASH]
        return types.SimpleNamespace(
            built=built, run=run_calls, parent=parent
        )

    def test_batch_per_task_models_reach_child_builder(self, harness):
        result = dt.delegate_task(
            tasks=[
                {"goal": "worker task A with a sufficiently long description", "model": FLASH},
                {"goal": "reviewer task B with a sufficiently long description", "model": QWEN},
            ],
            parent_agent=harness.parent,
        )
        assert '"error"' not in result, result
        assert len(harness.built) == 2
        assert harness.built[0]["model"] == FLASH
        assert harness.built[1]["model"] == QWEN

    def test_top_level_model_applies_to_single_goal_form(self, harness):
        dt.delegate_task(
            goal="single goal with a sufficiently long description here",
            model=QWEN,
            parent_agent=harness.parent,
        )
        assert harness.built[0]["model"] == QWEN

    def test_no_model_param_keeps_delegation_model(self, harness):
        dt.delegate_task(
            goal="single goal with a sufficiently long description here",
            parent_agent=harness.parent,
        )
        assert harness.built[0]["model"] == FLASH

    def test_invalid_top_level_model_rejected_before_spawn(self, harness):
        result = dt.delegate_task(
            goal="single goal with a sufficiently long description here",
            model="gpt-5",
            parent_agent=harness.parent,
        )
        assert "gpt-5" in _err_msg(result)
        assert harness.built == []

    def test_invalid_task_model_rejected_before_spawn(self, harness):
        result = dt.delegate_task(
            tasks=[
                {"goal": "worker task A with a sufficiently long description", "model": "gpt-5"},
                {"goal": "worker task B with a sufficiently long description"},
            ],
            parent_agent=harness.parent,
        )
        assert "gpt-5" in _err_msg(result)
        assert harness.built == []
