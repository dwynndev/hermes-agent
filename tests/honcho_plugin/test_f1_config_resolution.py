"""F1: Config Resolution & Provider Activation — comprehensive tests.

Tests the honcho plugin's config parsing, resolution chain, and activation logic.
Covers: camelCase→snake_case mapping, host-block precedence, type coercion,
edge cases (missing keys, wrong types, empty values), and security (key leakage).
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    _is_local_base_url,
    _normalize_recall_mode,
    _parse_context_tokens,
    _parse_dialectic_depth,
    _parse_dialectic_depth_levels,
    _parse_float_config,
    _parse_int_config,
    _resolve_bool,
)


# ---------------------------------------------------------------------------
# Unit tests: parsing helpers
# ---------------------------------------------------------------------------


class TestNormalizeRecallMode:
    """_normalize_recall_mode: legacy alias + validation."""

    def test_valid_modes_pass_through(self):
        for mode in ("hybrid", "context", "tools"):
            assert _normalize_recall_mode(mode) == mode

    def test_legacy_auto_maps_to_hybrid(self):
        assert _normalize_recall_mode("auto") == "hybrid"

    def test_invalid_mode_defaults_to_hybrid(self):
        assert _normalize_recall_mode("bogus") == "hybrid"
        assert _normalize_recall_mode("") == "hybrid"
        assert _normalize_recall_mode("TOOLS") == "hybrid"  # case-sensitive

    def test_none_coerced_to_hybrid(self):
        # None is not a str, but the function should handle it gracefully
        # via the dict .get() returning None → not in valid set → "hybrid"
        assert _normalize_recall_mode(None) == "hybrid"


class TestResolveBool:
    """_resolve_bool: first non-None wins, else default."""

    def test_first_non_none_wins(self):
        assert _resolve_bool(True, False, default=False) is True
        assert _resolve_bool(False, True, default=True) is False

    def test_skips_none_values(self):
        assert _resolve_bool(None, True, default=False) is True
        assert _resolve_bool(None, None, True, default=False) is True

    def test_all_none_returns_default(self):
        assert _resolve_bool(None, None, default=True) is True
        assert _resolve_bool(None, None, default=False) is False

    def test_truthy_coercion(self):
        # Non-bool truthy values get coerced via bool()
        assert _resolve_bool(1, None, default=False) is True
        assert _resolve_bool(0, None, default=True) is False
        assert _resolve_bool("", None, default=True) is False


class TestParseIntConfig:
    """_parse_int_config: host wins, root fallback, default."""

    def test_host_wins_over_root(self):
        assert _parse_int_config(5, 10, default=1) == 5

    def test_root_fallback_when_host_none(self):
        assert _parse_int_config(None, 7, default=1) == 7

    def test_default_when_both_none(self):
        assert _parse_int_config(None, None, default=42) == 42

    def test_string_coercion(self):
        assert _parse_int_config("3", None, default=1) == 3

    def test_invalid_string_falls_through(self):
        assert _parse_int_config("abc", "5", default=1) == 5
        assert _parse_int_config("abc", "xyz", default=9) == 9

    def test_float_truncated(self):
        assert _parse_int_config(3.9, None, default=1) == 3


class TestParseFloatConfig:
    """_parse_float_config: clamped ≥ 0."""

    def test_negative_clamped_to_zero(self):
        assert _parse_float_config(-5.0, None, default=1.0) == 0.0

    def test_host_wins(self):
        assert _parse_float_config(2.5, 9.9, default=1.0) == 2.5

    def test_invalid_falls_through(self):
        assert _parse_float_config("bad", "3.14", default=1.0) == 3.14


class TestParseContextTokens:
    """_parse_context_tokens: None means uncapped."""

    def test_host_wins(self):
        assert _parse_context_tokens(5000, 10000) == 5000

    def test_root_fallback(self):
        assert _parse_context_tokens(None, 8000) == 8000

    def test_both_none_returns_none(self):
        assert _parse_context_tokens(None, None) is None

    def test_invalid_returns_none(self):
        assert _parse_context_tokens("abc", "xyz") is None


class TestParseDialecticDepth:
    """_parse_dialectic_depth: clamped 1-3."""

    def test_valid_range(self):
        assert _parse_dialectic_depth(1, None) == 1
        assert _parse_dialectic_depth(2, None) == 2
        assert _parse_dialectic_depth(3, None) == 3

    def test_clamped_high(self):
        assert _parse_dialectic_depth(10, None) == 3

    def test_clamped_low(self):
        assert _parse_dialectic_depth(0, None) == 1
        assert _parse_dialectic_depth(-1, None) == 1

    def test_host_wins(self):
        assert _parse_dialectic_depth(2, 3) == 2

    def test_default_when_both_none(self):
        assert _parse_dialectic_depth(None, None) == 1


class TestParseDialecticDepthLevels:
    """_parse_dialectic_depth_levels: validates, truncates, pads."""

    def test_valid_levels(self):
        result = _parse_dialectic_depth_levels(["minimal", "low", "medium"], None, 3)
        assert result == ["minimal", "low", "medium"]

    def test_invalid_level_replaced_with_low(self):
        result = _parse_dialectic_depth_levels(["bogus", "high"], None, 2)
        assert result == ["low", "high"]

    def test_truncated_to_depth(self):
        result = _parse_dialectic_depth_levels(["minimal", "low", "high", "max"], None, 2)
        assert result == ["minimal", "low"]

    def test_padded_with_low(self):
        result = _parse_dialectic_depth_levels(["max"], None, 3)
        assert result == ["max", "low", "low"]

    def test_none_returns_none(self):
        assert _parse_dialectic_depth_levels(None, None, 2) is None

    def test_non_list_returns_none(self):
        assert _parse_dialectic_depth_levels("minimal", None, 2) is None


class TestIsLocalBaseUrl:
    """_is_local_base_url: loopback, private, CGNAT detection."""

    def test_localhost(self):
        assert _is_local_base_url("http://localhost:8000") is True
        assert _is_local_base_url("http://127.0.0.1:8000") is True
        assert _is_local_base_url("http://[::1]:8000") is True

    def test_private_ips(self):
        assert _is_local_base_url("http://192.168.1.100:8000") is True
        assert _is_local_base_url("http://10.0.0.5:8000") is True
        assert _is_local_base_url("http://172.16.0.1:8000") is True

    def test_cgnat_tailscale(self):
        assert _is_local_base_url("http://100.64.0.1:8000") is True
        assert _is_local_base_url("http://100.127.255.255:8000") is True

    def test_public_ips(self):
        assert _is_local_base_url("http://8.8.8.8:8000") is False
        assert _is_local_base_url("https://api.honcho.dev") is False

    def test_none_and_empty(self):
        assert _is_local_base_url(None) is False
        assert _is_local_base_url("") is False


# ---------------------------------------------------------------------------
# Integration tests: from_global_config
# ---------------------------------------------------------------------------


class TestFromGlobalConfigResolution:
    """Config resolution chain and field precedence."""

    def test_host_block_wins_over_root(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "root-key",
            "workspace": "root-ws",
            "hosts": {
                "hermes": {
                    "apiKey": "host-key",
                    "workspace": "host-ws",
                    "enabled": True,
                }
            }
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.api_key == "host-key"
        assert cfg.workspace_id == "host-ws"

    def test_root_fallback_when_host_missing_field(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "root-key",
            "workspace": "root-ws",
            "hosts": {
                "hermes": {
                    "enabled": True,
                    # no apiKey here
                }
            }
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.api_key == "root-key"
        assert cfg.workspace_id == "root-ws"

    def test_env_fallback_for_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HONCHO_API_KEY", "env-key-123")
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "enabled": True,
            "hosts": {"hermes": {"enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.api_key == "env-key-123"

    def test_missing_file_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HONCHO_API_KEY", "env-only-key")
        config_path = tmp_path / "nonexistent.json"
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.api_key == "env-only-key"
        assert cfg.enabled is True

    def test_corrupt_json_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HONCHO_API_KEY", "env-fallback")
        config_path = tmp_path / "honcho.json"
        config_path.write_text("{invalid json!!!")
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.api_key == "env-fallback"

    def test_auto_enable_with_api_key_no_explicit_enabled(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "some-key"}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.enabled is True

    def test_auto_enable_with_base_url_no_explicit_enabled(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"baseUrl": "http://localhost:8000"}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.enabled is True

    def test_no_key_no_url_no_enabled_is_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"workspace": "test"}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.enabled is False

    def test_explicit_enabled_false_overrides_auto(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "key-exists",
            "enabled": False,
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.enabled is False

    def test_host_enabled_wins_over_root(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "enabled": True,
            "hosts": {"hermes": {"enabled": False}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.enabled is False


class TestFromGlobalConfigOptimizationKnobs:
    """Phase 1 optimization knobs parse correctly."""

    def _make_config(self, host_overrides=None, root_overrides=None):
        """Helper: build a config dict with optimization knobs."""
        root = {"apiKey": "test-key", "enabled": True}
        if root_overrides:
            root.update(root_overrides)
        host = {"enabled": True}
        if host_overrides:
            host.update(host_overrides)
        root["hosts"] = {"hermes": host}
        return root

    def test_recall_mode_tools(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"recallMode": "tools"}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.recall_mode == "tools"

    def test_recall_mode_legacy_auto(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"recallMode": "auto"}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.recall_mode == "hybrid"

    def test_context_cadence(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"contextCadence": 10}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.context_cadence == 10

    def test_dialectic_cadence(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"dialecticCadence": 20}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.dialectic_cadence == 20

    def test_save_messages_false(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"saveMessages": False}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.save_messages is False

    def test_write_frequency_session(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"writeFrequency": "session"}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.write_frequency == "session"

    def test_write_frequency_integer(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"writeFrequency": 5}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.write_frequency == 5

    def test_dialectic_reasoning_level_minimal(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"dialecticReasoningLevel": "minimal"}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.dialectic_reasoning_level == "minimal"

    def test_dialectic_depth_clamped(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={"dialecticDepth": 99}
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.dialectic_depth == 3

    def test_full_optimization_config(self, tmp_path):
        """Verify the exact Phase 1 config we deployed."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps(self._make_config(
            host_overrides={
                "recallMode": "tools",
                "contextCadence": 10,
                "dialecticCadence": 20,
                "dialecticDepth": 1,
                "dialecticReasoningLevel": "minimal",
                "saveMessages": False,
                "writeFrequency": "session",
                "dialecticMaxChars": 400,
                "messageMaxChars": 15000,
            }
        )))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.recall_mode == "tools"
        assert cfg.context_cadence == 10
        assert cfg.dialectic_cadence == 20
        assert cfg.dialectic_depth == 1
        assert cfg.dialectic_reasoning_level == "minimal"
        assert cfg.save_messages is False
        assert cfg.write_frequency == "session"
        assert cfg.dialectic_max_chars == 400
        assert cfg.message_max_chars == 15000


class TestFromGlobalConfigMultiHost:
    """Multi-host config (hermes, hermes_builder, hermes_qa, hermes_researcher)."""

    def test_each_host_gets_own_config(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "shared-key",
            "enabled": True,
            "hosts": {
                "hermes": {"aiPeer": "hermes", "recallMode": "tools"},
                "hermes_builder": {"aiPeer": "builder", "recallMode": "hybrid"},
                "hermes_qa": {"aiPeer": "qa", "recallMode": "context"},
            }
        }))
        cfg_h = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        cfg_b = HonchoClientConfig.from_global_config(host="hermes_builder", config_path=config_path)
        cfg_q = HonchoClientConfig.from_global_config(host="hermes_qa", config_path=config_path)

        assert cfg_h.ai_peer == "hermes"
        assert cfg_h.recall_mode == "tools"
        assert cfg_b.ai_peer == "builder"
        assert cfg_b.recall_mode == "hybrid"
        assert cfg_q.ai_peer == "qa"
        assert cfg_q.recall_mode == "context"

    def test_unknown_host_gets_root_defaults(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "root-key",
            "workspace": "root-ws",
            "enabled": True,
            "hosts": {
                "hermes": {"workspace": "host-ws"},
            }
        }))
        cfg = HonchoClientConfig.from_global_config(host="nonexistent", config_path=config_path)
        # Unknown host → empty host block → root values
        assert cfg.api_key == "root-key"
        assert cfg.workspace_id == "root-ws"


class TestFromGlobalConfigEdgeCases:
    """Edge cases: wrong types, empty values, boundary conditions."""

    def test_empty_json_object(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        config_path = tmp_path / "honcho.json"
        config_path.write_text("{}")
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.enabled is False
        assert cfg.api_key is None

    def test_cadence_as_string(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k",
            "enabled": True,
            "hosts": {"hermes": {"contextCadence": "5", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.context_cadence == 5

    def test_cadence_as_invalid_string(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k",
            "enabled": True,
            "hosts": {"hermes": {"contextCadence": "abc", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        # Invalid → falls through to default (1)
        assert cfg.context_cadence == 1

    def test_base_url_alias(self, tmp_path):
        """Both baseUrl and base_url should work."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "base_url": "http://localhost:9999",
            "enabled": True,
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.base_url == "http://localhost:9999"

    def test_save_messages_false_not_overridden_by_default(self, tmp_path):
        """False is a valid value — must not be treated as 'not set'."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k",
            "enabled": True,
            "saveMessages": True,
            "hosts": {"hermes": {"saveMessages": False, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.save_messages is False


class TestSecurityNoKeyLeakage:
    """API key must not appear in logs, repr, or error messages.

    BUG FOUND (2026-08-02): HonchoClientConfig is a plain dataclass whose
    auto-generated __repr__ includes api_key in cleartext. Any logger.debug
    or traceback that prints the config object leaks the key.
    Fix: add __repr__ that redacts api_key, or use field(repr=False).
    """

    @pytest.mark.xfail(strict=True, reason="REAL BUG: dataclass repr leaks api_key in cleartext")
    def test_repr_does_not_contain_key(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "sk-super-secret-key-12345",
            "enabled": True,
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        repr_str = repr(cfg)
        assert "sk-super-secret-key-12345" not in repr_str

    @pytest.mark.xfail(strict=True, reason="REAL BUG: dataclass str leaks api_key in cleartext")
    def test_str_does_not_contain_key(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "sk-super-secret-key-12345",
            "enabled": True,
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        str_repr = str(cfg)
        assert "sk-super-secret-key-12345" not in str_repr
