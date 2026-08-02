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
    _normalize_observation_mode,
    _parse_context_tokens,
    _parse_dialectic_depth,
    _parse_dialectic_depth_levels,
    _parse_float_config,
    _parse_int_config,
    _parse_string_map,
    _parse_optional_string,
    _resolve_bool,
    _resolve_observation,
    _resolve_optional_float,
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


class TestNormalizeObservationMode:
    """_normalize_observation_mode: aliases + validation."""

    def test_valid_modes(self):
        assert _normalize_observation_mode("directional") == "directional"
        assert _normalize_observation_mode("unified") == "unified"

    def test_aliases(self):
        assert _normalize_observation_mode("shared") == "unified"
        assert _normalize_observation_mode("separate") == "directional"
        assert _normalize_observation_mode("cross") == "directional"

    def test_invalid_defaults_to_directional(self):
        assert _normalize_observation_mode("bogus") == "directional"
        assert _normalize_observation_mode("") == "directional"


class TestResolveObservation:
    """_resolve_observation: preset + granular override."""

    def test_directional_preset_all_true(self):
        result = _resolve_observation("directional", None)
        assert result == {
            "user_observe_me": True, "user_observe_others": True,
            "ai_observe_me": True, "ai_observe_others": True,
        }

    def test_unified_preset(self):
        result = _resolve_observation("unified", None)
        assert result == {
            "user_observe_me": True, "user_observe_others": False,
            "ai_observe_me": False, "ai_observe_others": True,
        }

    def test_granular_overrides_preset(self):
        obs = {"user": {"observeMe": False}, "ai": {"observeOthers": False}}
        result = _resolve_observation("directional", obs)
        assert result["user_observe_me"] is False
        assert result["user_observe_others"] is True  # preset default
        assert result["ai_observe_me"] is True  # preset default
        assert result["ai_observe_others"] is False

    def test_empty_observation_uses_preset(self):
        result = _resolve_observation("unified", {})
        assert result["ai_observe_me"] is False

    def test_invalid_mode_falls_to_directional(self):
        result = _resolve_observation("nonexistent", None)
        assert result["user_observe_me"] is True
        assert result["ai_observe_me"] is True


class TestParseStringMap:
    """_parse_string_map: host whole-map override, string coercion."""

    def test_host_map_wins_whole(self):
        host = {"userPeerAliases": {"111": "alice"}}
        root = {"userPeerAliases": {"222": "bob"}}
        result = _parse_string_map(host, root, "userPeerAliases")
        assert result == {"111": "alice"}

    def test_root_fallback(self):
        host = {}
        root = {"userPeerAliases": {"333": "carol"}}
        result = _parse_string_map(host, root, "userPeerAliases")
        assert result == {"333": "carol"}

    def test_non_dict_returns_empty(self):
        result = _parse_string_map({"userPeerAliases": "bad"}, {}, "userPeerAliases")
        assert result == {}

    def test_strips_whitespace(self):
        host = {"userPeerAliases": {" 111 ": " alice "}}
        result = _parse_string_map(host, {}, "userPeerAliases")
        assert result == {"111": "alice"}

    def test_empty_values_skipped(self):
        host = {"userPeerAliases": {"111": "", "222": "bob"}}
        result = _parse_string_map(host, {}, "userPeerAliases")
        assert result == {"222": "bob"}

    def test_none_value_becomes_empty_and_skipped(self):
        host = {"userPeerAliases": {"111": None, "222": "bob"}}
        result = _parse_string_map(host, {}, "userPeerAliases")
        assert result == {"222": "bob"}


class TestParseOptionalString:
    """_parse_optional_string: host empty string overrides root."""

    def test_host_wins(self):
        result = _parse_optional_string({"key": "host"}, {"key": "root"}, "key")
        assert result == "host"

    def test_host_empty_string_overrides_root(self):
        result = _parse_optional_string({"key": ""}, {"key": "root"}, "key")
        assert result == ""

    def test_root_fallback(self):
        result = _parse_optional_string({}, {"key": "root"}, "key")
        assert result == "root"

    def test_default_when_missing(self):
        result = _parse_optional_string({}, {}, "key", default="def")
        assert result == "def"

    def test_none_becomes_default(self):
        result = _parse_optional_string({"key": None}, {}, "key", default="def")
        assert result == "def"

    def test_strips_whitespace(self):
        result = _parse_optional_string({"key": "  val  "}, {}, "key")
        assert result == "val"


class TestResolveOptionalFloat:
    """_resolve_optional_float: first positive wins."""

    def test_first_positive_wins(self):
        assert _resolve_optional_float(2.5, 9.9) == 2.5

    def test_skips_none(self):
        assert _resolve_optional_float(None, 3.14) == 3.14

    def test_skips_zero_and_negative(self):
        assert _resolve_optional_float(0, -1, 5.0) == 5.0

    def test_string_coercion(self):
        assert _resolve_optional_float("2.5") == 2.5

    def test_empty_string_skipped(self):
        assert _resolve_optional_float("", "3.0") == 3.0

    def test_invalid_string_skipped(self):
        assert _resolve_optional_float("abc", "4.0") == 4.0

    def test_all_invalid_returns_none(self):
        assert _resolve_optional_float(None, "abc", 0) is None


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


class TestFromGlobalConfigGatewayFields:
    """Gateway identity fields: pinUserPeer, userPeerAliases, runtimePeerPrefix."""

    def test_pin_user_peer_host_wins(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "pinUserPeer": False,
            "hosts": {"hermes": {"pinUserPeer": True, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.pin_peer_name is True

    def test_pin_peer_name_legacy_alias(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"pinPeerName": True, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.pin_peer_name is True

    def test_user_peer_aliases(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {
                "userPeerAliases": {"7654321": "alice", "9999": "bob"},
                "enabled": True
            }}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.user_peer_aliases == {"7654321": "alice", "9999": "bob"}

    def test_runtime_peer_prefix(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"runtimePeerPrefix": "telegram_", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.runtime_peer_prefix == "telegram_"

    def test_default_gateway_fields(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "k", "enabled": True}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.pin_peer_name is False
        assert cfg.user_peer_aliases == {}
        assert cfg.runtime_peer_prefix == ""


class TestFromGlobalConfigObservation:
    """Observation mode and per-peer toggles via from_global_config."""

    def test_default_new_install_is_directional(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"baseUrl": "http://localhost:8000"}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        # New install (no host block, no explicit config) → directional
        assert cfg.observation_mode == "directional"
        assert cfg.user_observe_me is True
        assert cfg.ai_observe_me is True

    def test_explicitly_configured_defaults_to_unified(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        # Explicitly configured (host block exists) → unified (migration guard)
        assert cfg.observation_mode == "unified"

    def test_explicit_observation_mode_directional(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"observationMode": "directional", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.observation_mode == "directional"
        assert cfg.user_observe_me is True
        assert cfg.user_observe_others is True
        assert cfg.ai_observe_me is True
        assert cfg.ai_observe_others is True

    def test_granular_observation_overrides(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {
                "observationMode": "directional",
                "observation": {"ai": {"observeMe": False}},
                "enabled": True
            }}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.ai_observe_me is False
        assert cfg.ai_observe_others is True  # preset default kept


class TestFromGlobalConfigMiscFields:
    """Remaining config fields: timeout, dialectic_dynamic, heuristic, etc."""

    def test_timeout_from_host(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"timeout": 15.5, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.timeout == 15.5

    def test_timeout_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HONCHO_TIMEOUT", "22.0")
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "k", "enabled": True}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.timeout == 22.0

    def test_request_timeout_alias(self, tmp_path):
        """requestTimeout is an alias for timeout."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"requestTimeout": 12.0, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.timeout == 12.0

    def test_dialectic_dynamic_false(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"dialecticDynamic": False, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.dialectic_dynamic is False

    def test_reasoning_heuristic_false(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"reasoningHeuristic": False, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.reasoning_heuristic is False

    def test_reasoning_level_cap(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"reasoningLevelCap": "max", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.reasoning_level_cap == "max"

    def test_dialectic_max_input_chars(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"dialecticMaxInputChars": 5000, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.dialectic_max_input_chars == 5000

    def test_injection_frequency(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"injectionFrequency": "first-turn", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.injection_frequency == "first-turn"

    def test_query_rewrite_true(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"queryRewrite": True, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.query_rewrite is True

    def test_first_turn_waits(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {
                "firstTurnBaseWait": 5.0,
                "firstTurnDialecticWait": 4.0,
                "enabled": True
            }}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.first_turn_base_wait == 5.0
        assert cfg.first_turn_dialectic_wait == 4.0

    def test_session_strategy(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"sessionStrategy": "global", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.session_strategy == "global"

    def test_context_tokens(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"contextTokens": 4096, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.context_tokens == 4096

    def test_peer_name_and_ai_peer(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "peerName": "eri",
            "hosts": {"hermes": {"aiPeer": "coder", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.peer_name == "eri"
        assert cfg.ai_peer == "coder"

    def test_init_on_session_start(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"initOnSessionStart": True, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.init_on_session_start is True

    def test_session_peer_prefix(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"sessionPeerPrefix": "sess_", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.session_peer_prefix == "sess_"

    def test_environment_field(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "environment": "local",
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.environment == "local"

    def test_base_url_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HONCHO_BASE_URL", "http://env-host:9000")
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "k", "enabled": True}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.base_url == "http://env-host:9000"

    def test_workspace_defaults_to_host_name(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "k", "enabled": True}))
        cfg = HonchoClientConfig.from_global_config(host="myhost", config_path=config_path)
        assert cfg.workspace_id == "myhost"

    def test_ai_peer_defaults_to_host_name(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "k", "enabled": True}))
        cfg = HonchoClientConfig.from_global_config(host="myhost", config_path=config_path)
        assert cfg.ai_peer == "myhost"

    def test_dialectic_depth_levels_integration(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {
                "dialecticDepth": 3,
                "dialecticDepthLevels": ["minimal", "low", "high"],
                "enabled": True
            }}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.dialectic_depth == 3
        assert cfg.dialectic_depth_levels == ["minimal", "low", "high"]

    def test_defaults_for_all_misc_fields(self, tmp_path):
        """Verify all defaults when only apiKey is set."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "k", "enabled": True}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.timeout is None
        assert cfg.dialectic_dynamic is True
        assert cfg.reasoning_heuristic is True
        assert cfg.reasoning_level_cap == "high"
        assert cfg.dialectic_max_input_chars == 10000
        assert cfg.injection_frequency == "every-turn"
        assert cfg.query_rewrite is False
        assert cfg.first_turn_base_wait == 3.0
        assert cfg.first_turn_dialectic_wait == 2.0
        assert cfg.session_strategy == "per-directory"
        assert cfg.context_tokens is None
        assert cfg.dialectic_max_chars == 600
        assert cfg.message_max_chars == 25000
        assert cfg.dialectic_reasoning_level == "low"
        assert cfg.recall_mode == "hybrid"
        assert cfg.context_cadence == 1
        assert cfg.dialectic_cadence == 1
        assert cfg.init_on_session_start is False
        assert cfg.environment == "production"


class TestRealBugsFound:
    """Tests documenting REAL BUGS in the source code.

    These are pinned as xfail(strict=True) so they fail the build
    the moment the bug is fixed (prompting marker removal).
    """

    @pytest.mark.xfail(strict=True, reason="REAL BUG: host-level baseUrl ignored — source lines 536-541 only read raw, not host_block")
    def test_host_base_url_should_win(self, tmp_path):
        """Host-level baseUrl should override root, like every other field."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "baseUrl": "http://root:8000",
            "hosts": {"hermes": {"baseUrl": "http://host:9000", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.base_url == "http://host:9000"

    @pytest.mark.xfail(strict=True, reason="REAL BUG: writeFrequency=0 swallowed by `or` chain (0 is falsy)")
    def test_write_frequency_zero_should_be_valid(self, tmp_path):
        """writeFrequency=0 means 'never auto-write' — valid but swallowed."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"writeFrequency": 0, "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        assert cfg.write_frequency == 0

    def test_dialectic_reasoning_level_not_validated(self, tmp_path):
        """DOCUMENTED GAP: any string passes through without validation.
        Unlike recall_mode and observation_mode, no normalization occurs."""
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "k", "enabled": True,
            "hosts": {"hermes": {"dialecticReasoningLevel": "BOGUS", "enabled": True}}
        }))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=config_path)
        # This passes — documenting the gap (no validation)
        assert cfg.dialectic_reasoning_level == "BOGUS"


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
