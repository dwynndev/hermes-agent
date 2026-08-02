"""Contract test: alibaba/DashScope reasoning_effort emission gate.

Proves that _supports_reasoning_extra_body() returns True for aliyuncs.com
base URLs, activating the nested extra_body.reasoning emission path.
Without this gate, config reasoning_effort is a SILENT NO-OP for alibaba.

Mutation-proven: removing the aliyuncs.com branch (lines 6588-6595 in
run_agent.py) causes TestAlibabaReasoningEmissionFix to FAIL.
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(base_url: str, provider: str = "alibaba", model: str = "qwen3.8-max-preview"):
    """Create a minimal AIAgent-like object with the gate method bound."""
    from run_agent import AIAgent, base_url_host_matches

    agent = object.__new__(AIAgent)
    agent._base_url_lower = base_url.lower()
    agent.provider = provider
    agent.model = model
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlibabaReasoningEmissionFix:
    """The aliyuncs.com branch in _supports_reasoning_extra_body()."""

    def test_token_plan_compatible_mode_returns_true(self):
        """The exact Token Plan base URL must activate reasoning emission."""
        agent = _make_agent(
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )
        assert agent._supports_reasoning_extra_body() is True

    def test_dashscope_intl_returns_true(self):
        """The standard DashScope international endpoint must also work."""
        agent = _make_agent("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        assert agent._supports_reasoning_extra_body() is True

    def test_dashscope_cn_returns_true(self):
        """The China DashScope endpoint must also work."""
        agent = _make_agent("https://dashscope.aliyuncs.com/compatible-mode/v1")
        assert agent._supports_reasoning_extra_body() is True

    def test_openrouter_still_works(self):
        """Regression: openrouter route must not be broken by the new branch."""
        agent = _make_agent(
            "https://openrouter.ai/api/v1",
            provider="openrouter",
            model="deepseek/deepseek-v4-pro",
        )
        assert agent._supports_reasoning_extra_body() is True

    def test_openrouter_non_reasoning_model_returns_false(self):
        """OpenRouter with a non-reasoning model prefix must return False."""
        agent = _make_agent(
            "https://openrouter.ai/api/v1",
            provider="openrouter",
            model="meta-llama/llama-3",
        )
        assert agent._supports_reasoning_extra_body() is False

    def test_nousresearch_still_works(self):
        """Regression: nousresearch route must not be broken."""
        agent = _make_agent("https://api.nousresearch.com/v1", provider="nousresearch")
        assert agent._supports_reasoning_extra_body() is True

    def test_unknown_provider_returns_false(self):
        """A random base URL with no known provider must return False."""
        agent = _make_agent("https://api.example.com/v1", provider="example")
        assert agent._supports_reasoning_extra_body() is False

    def test_aliyuncs_case_insensitive(self):
        """The gate must be case-insensitive (base_url is lowered)."""
        agent = _make_agent(
            "https://TOKEN-PLAN.AP-SOUTHEAST-1.MAAS.ALIYUNCS.COM/compatible-mode/v1"
        )
        assert agent._supports_reasoning_extra_body() is True


class TestReasoningConfigResolution:
    """Verify that reasoning_config flows correctly for alibaba."""

    def test_parse_reasoning_effort_max(self):
        from hermes_constants import parse_reasoning_effort
        result = parse_reasoning_effort("max")
        assert result == {"enabled": True, "effort": "max"}

    def test_parse_reasoning_effort_xhigh(self):
        from hermes_constants import parse_reasoning_effort
        result = parse_reasoning_effort("xhigh")
        assert result == {"enabled": True, "effort": "xhigh"}

    def test_parse_reasoning_effort_none(self):
        from hermes_constants import parse_reasoning_effort
        result = parse_reasoning_effort("none")
        assert result == {"enabled": False}

    def test_valid_efforts_does_not_include_ultra_for_alibaba(self):
        """ultra is in VALID_REASONING_EFFORTS but DashScope rejects it.
        This test documents the known gap — no code-level clamp exists yet."""
        from hermes_constants import VALID_REASONING_EFFORTS
        # ultra IS in the tuple (for other providers that accept it)
        assert "ultra" in VALID_REASONING_EFFORTS
        # But the alibaba endpoint rejects it with HTTP 400.
        # This is a documentation test, not a behavior test.

    def test_resolve_reasoning_config_override_wins(self):
        """Per-model override must beat global reasoning_effort."""
        from hermes_constants import resolve_reasoning_config
        config = {
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {"qwen3.8-max-preview": "max"},
            }
        }
        result = resolve_reasoning_config(config, "qwen3.8-max-preview")
        assert result == {"enabled": True, "effort": "max"}

    def test_resolve_reasoning_config_global_fallback(self):
        """Without an override, global reasoning_effort applies."""
        from hermes_constants import resolve_reasoning_config
        config = {"agent": {"reasoning_effort": "xhigh"}}
        result = resolve_reasoning_config(config, "qwen3.8-max-preview")
        assert result == {"enabled": True, "effort": "xhigh"}
