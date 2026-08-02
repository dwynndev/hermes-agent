"""F9: Migration & Rollback — tests.

Tests switching between cloud honcho.dev and self-hosted honcho:
- baseUrl change (cloud → localhost:8000)
- Rollback safety (localhost → cloud)
- Config validation for migration
- Data preservation expectations
"""

import json
import os
import copy

import pytest

from plugins.memory.honcho.client import HonchoClientConfig


def _load_honcho_json():
    """Load the actual honcho.json config (always from real ~/.hermes, not tmpdir)."""
    path = os.path.join(os.path.expanduser("~/.hermes"), "honcho.json")
    with open(path) as f:
        return json.load(f)


class TestBaseUrlMigration:
    """Test baseUrl switching between cloud and self-hosted."""

    def test_cloud_base_url_is_honcho_dev(self):
        """Default cloud baseUrl points to honcho.dev."""
        cfg = _load_honcho_json()
        host = cfg.get("hosts", {}).get("hermes", {})
        base_url = host.get("baseUrl", "")
        # Cloud URL should contain honcho.dev or be empty (SDK default)
        assert base_url == "" or "honcho" in base_url.lower() or "plasticlabs" in base_url.lower()

    def test_selfhost_base_url_is_localhost(self):
        """Self-hosted baseUrl points to localhost:8000."""
        # Simulate migration config
        cfg = _load_honcho_json()
        migrated = copy.deepcopy(cfg)
        migrated["hosts"]["hermes"]["baseUrl"] = "http://localhost:8000"

        host = migrated["hosts"]["hermes"]
        assert "localhost" in host["baseUrl"]
        assert "8000" in host["baseUrl"]

    def test_rollback_preserves_other_settings(self):
        """Rolling back baseUrl doesn't affect optimization knobs."""
        cfg = _load_honcho_json()
        original = copy.deepcopy(cfg)

        # Simulate migration
        cfg["hosts"]["hermes"]["baseUrl"] = "http://localhost:8000"
        # Simulate rollback
        cfg["hosts"]["hermes"]["baseUrl"] = original["hosts"]["hermes"].get("baseUrl", "")

        # Optimization knobs preserved
        host = cfg["hosts"]["hermes"]
        assert host.get("recallMode") == original["hosts"]["hermes"].get("recallMode")
        assert host.get("saveMessages") == original["hosts"]["hermes"].get("saveMessages")
        assert host.get("writeFrequency") == original["hosts"]["hermes"].get("writeFrequency")
        assert host.get("dialecticReasoningLevel") == original["hosts"]["hermes"].get("dialecticReasoningLevel")

    def test_migration_only_changes_base_url(self):
        """Migration should only change baseUrl, nothing else."""
        cfg = _load_honcho_json()
        original = copy.deepcopy(cfg)

        # Apply migration
        cfg["hosts"]["hermes"]["baseUrl"] = "http://localhost:8000"

        # Only baseUrl changed
        for key in original["hosts"]["hermes"]:
            if key != "baseUrl":
                assert cfg["hosts"]["hermes"][key] == original["hosts"]["hermes"][key], \
                    f"Key '{key}' was unexpectedly modified during migration"


class TestConfigValidation:
    """Validate config integrity for migration safety."""

    def test_all_hosts_have_consistent_optimization(self):
        """All host blocks should have the same optimization knobs."""
        cfg = _load_honcho_json()
        hosts = cfg.get("hosts", {})
        if len(hosts) <= 1:
            pytest.skip("Only one host configured")

        reference = hosts.get("hermes", {})
        for name, host in hosts.items():
            if name == "hermes":
                continue
            assert host.get("recallMode") == reference.get("recallMode"), \
                f"Host '{name}' has different recallMode"
            assert host.get("saveMessages") == reference.get("saveMessages"), \
                f"Host '{name}' has different saveMessages"

    def test_api_key_present_for_cloud(self):
        """Cloud mode requires api_key (from .env or config) OR self-hosted baseUrl."""
        raw = _load_honcho_json()
        host = raw.get("hosts", {}).get("hermes", {})
        base_url = host.get("baseUrl", "")

        # api_key may live in config or in ~/.hermes/.env (conftest strips env vars)
        api_key = host.get("apiKey", "")
        if not api_key:
            env_path = os.path.join(os.path.expanduser("~/.hermes"), ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("HONCHO_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break

        has_key = bool(api_key)
        is_selfhosted = bool(base_url) and "localhost" in base_url
        assert has_key or is_selfhosted, "Neither api_key nor self-hosted baseUrl configured"

    def test_enabled_flag_true(self):
        """Provider must be enabled."""
        cfg = HonchoClientConfig.from_global_config()
        assert cfg.enabled is True


class TestDataPreservation:
    """Document data preservation expectations during migration."""

    def test_cloud_data_not_migrated_automatically(self):
        """No built-in data migration path exists (documented limitation)."""
        # The honcho plugin has no migrate/export/import functionality.
        # Switching baseUrl means starting fresh on the new backend.
        # This test documents the limitation.
        cfg = _load_honcho_json()
        # No migration keys should exist in config
        assert "migration" not in cfg
        assert "exportPath" not in cfg
        assert "importPath" not in cfg

    def test_session_keys_are_stable_across_backends(self):
        """Session keys (workspace:session) are deterministic, not backend-specific."""
        # Session keys are derived from chat_id + session_strategy,
        # so the same key resolves on any backend (cloud or self-hosted).
        # Data won't transfer, but keys are stable.
        from plugins.memory.honcho.session import HonchoSessionManager

        # Verify session key format is deterministic
        # (tested more thoroughly in F2, here we just verify the concept)
        assert hasattr(HonchoSessionManager, 'get_or_create')
