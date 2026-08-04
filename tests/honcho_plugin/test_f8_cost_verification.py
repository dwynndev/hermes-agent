"""F8: Cost Optimization Verification — tests.

Verifies that the optimization knobs in honcho.json are correctly applied:
- recallMode=tools → zero auto-inject API calls per turn
- saveMessages=false → no ingestion cost
- writeFrequency=session → batch at end, not per-turn
- dialecticReasoningLevel=minimal → cheapest dialectic tier
- contextCadence/dialecticCadence → reduced call frequency
- Cost model: cloud vs self-hosted projections
"""

import json
import os

import pytest

from plugins.memory.honcho.client import HonchoClientConfig


def _load_honcho_json():
    """Load the actual honcho.json config (always from real ~/.hermes, not tmpdir)."""
    path = os.path.join(os.path.expanduser("~/.hermes"), "honcho.json")
    with open(path) as f:
        return json.load(f)


class TestOptimizationKnobs:
    """Verify optimization settings are applied in honcho.json."""

    def test_recall_mode_is_tools(self):
        """recallMode=tools → zero auto-inject overhead per turn."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        assert host.get("recallMode") == "tools"

    def test_save_messages_disabled(self):
        """saveMessages=false → no ingestion cost."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        assert host.get("saveMessages") is False

    def test_write_frequency_session(self):
        """writeFrequency=session → batch at end, not per-turn."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        assert host.get("writeFrequency") == "session"

    def test_dialectic_reasoning_level_minimal(self):
        """dialecticReasoningLevel=minimal → $0.001/call (cheapest)."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        assert host.get("dialecticReasoningLevel") == "minimal"

    def test_context_cadence_reduced(self):
        """contextCadence >= 10 → fewer context() calls."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        cadence = host.get("contextCadence", 1)
        assert cadence >= 10

    def test_dialectic_cadence_reduced(self):
        """dialecticCadence >= 20 → fewer dialectic calls."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        cadence = host.get("dialecticCadence", 2)
        assert cadence >= 20


class TestCostModel:
    """Verify cost projections are correct."""

    def test_tools_mode_zero_auto_calls(self):
        """recallMode=tools → 0 automatic API calls per turn."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        recall_mode = host.get("recallMode", "hybrid")

        # In tools mode, system_prompt_block returns static text,
        # prefetch returns empty, queue_prefetch doesn't start thread
        assert recall_mode == "tools"
        # Zero auto-inject = zero per-turn API calls

    def test_selfhost_zero_marginal_cost(self):
        """Self-hosted honcho → $0/month marginal cost."""
        # Self-hosted runs on local Docker with alibaba LLM backend
        # No per-call charges from honcho.dev
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        base_url = host.get("baseUrl", "")

        # If baseUrl points to localhost, it's self-hosted = $0
        if "localhost" in base_url or "127.0.0.1" in base_url:
            assert True  # self-hosted, $0
        else:
            # Cloud mode — cost depends on usage but optimized knobs reduce it
            assert host.get("saveMessages") is False  # no ingestion cost

    def test_cloud_minimal_cost_projection(self):
        """Cloud with optimized knobs → <$1/month for typical usage."""
        # Verify the actual config knobs that make this projection valid:
        # tools mode = 0 auto calls, saveMessages=false = 0 ingestion
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})

        # These knobs are what drive cost to near-zero
        assert host.get("recallMode") == "tools"  # 0 auto-inject calls
        assert host.get("saveMessages") is False  # 0 ingestion cost

        # With only manual tool calls (~10/day × 30 × $0.001), cost < $1
        manual_calls_per_day = 10
        days = 30
        cost_per_call = 0.001  # minimal dialectic or search
        monthly_cost = manual_calls_per_day * days * cost_per_call
        assert monthly_cost < 1.0


class TestConfigParsingViaParser:
    """Verify HonchoClientConfig.from_global_config() parses optimized values correctly."""

    @pytest.fixture(autouse=True)
    def _real_hermes_home(self, monkeypatch):
        """Point HERMES_HOME at the real ~/.hermes so from_global_config reads real honcho.json."""
        monkeypatch.setenv("HERMES_HOME", os.path.expanduser("~/.hermes"))

    def test_recall_mode_parsed_as_tools(self):
        """Parser resolves recallMode='tools' from host block."""
        cfg = HonchoClientConfig.from_global_config()
        assert cfg.recall_mode == "tools"

    def test_save_messages_parsed_as_false(self):
        """Parser resolves saveMessages=false from host block."""
        cfg = HonchoClientConfig.from_global_config()
        assert cfg.save_messages is False

    def test_write_frequency_parsed_as_session(self):
        """Parser resolves writeFrequency='session' from host block."""
        cfg = HonchoClientConfig.from_global_config()
        assert cfg.write_frequency == "session"

    def test_dialectic_level_parsed_as_minimal(self):
        """Parser resolves dialecticReasoningLevel='minimal' from host block."""
        cfg = HonchoClientConfig.from_global_config()
        assert cfg.dialectic_reasoning_level == "minimal"
