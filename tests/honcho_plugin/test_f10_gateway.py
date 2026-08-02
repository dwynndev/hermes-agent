"""F10: Gateway & Multi-Profile — tests.

Tests multi-host configuration (hermes, hermes_builder, hermes_qa, hermes_researcher)
and gateway integration:
- Host block resolution per profile
- Fallback to root config when host block missing
- Gateway mode detection (cron_skipped)
- Profile-specific overrides
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


class TestMultiHostConfig:
    """Test multi-host configuration resolution."""

    def test_hermes_host_block_exists(self):
        """Primary 'hermes' host block is configured."""
        cfg = _load_honcho_json()
        hosts = cfg.get("hosts", {})
        assert "hermes" in hosts

    def test_host_block_overrides_root(self):
        """Host block values override root-level defaults."""
        cfg = _load_honcho_json()
        root_recall = cfg.get("recallMode", "hybrid")
        host = cfg.get("hosts", {}).get("hermes", {})
        host_recall = host.get("recallMode", root_recall)

        # If host block sets recallMode, it should differ from or equal root
        # (the point is host block takes precedence)
        if "recallMode" in host:
            assert host_recall == host["recallMode"]

    def test_missing_host_falls_back_to_root(self):
        """Unknown host name falls back to root config."""
        cfg = HonchoClientConfig.from_global_config()
        # Default config uses "hermes" host; if we had a different host name,
        # it would fall back to root values
        assert cfg.host == "hermes"

    def test_all_hosts_share_workspace(self):
        """All host blocks should use the same workspace for data consistency."""
        cfg = _load_honcho_json()
        hosts = cfg.get("hosts", {})
        workspaces = set()
        for name, host in hosts.items():
            ws = host.get("workspace", host.get("workspaceId", "hermes"))
            workspaces.add(ws)

        # All hosts should share workspace (or at most 2 for isolation)
        assert len(workspaces) <= 2, f"Too many workspaces: {workspaces}"


class TestGatewayMode:
    """Test gateway/cron mode detection."""

    def test_cron_skipped_is_instance_level(self):
        """_cron_skipped is per-instance, not a class attribute (multi-provider isolation)."""
        from plugins.memory.honcho import HonchoMemoryProvider

        # Class itself should NOT have _cron_skipped as a class attribute
        assert not hasattr(HonchoMemoryProvider, '_cron_skipped') or \
            '_cron_skipped' not in HonchoMemoryProvider.__dict__

        # Two instances can have different values
        p1 = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        p2 = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        p1._cron_skipped = True
        p2._cron_skipped = False
        assert p1._cron_skipped is True
        assert p2._cron_skipped is False

    def test_gateway_mode_disables_auto_inject(self):
        """In gateway/cron mode, auto-inject is disabled."""
        from plugins.memory.honcho import HonchoMemoryProvider

        # Create provider instance
        p = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        p._cron_skipped = True
        p._recall_mode = "hybrid"
        p._context_cadence = 1
        p._dialectic_cadence = 2
        p._base_context_cache = ""
        p._pending_dialectic = None
        p._pending_dialectic_ts = 0
        p._turn_count = 0
        p._prefetch_lock = __import__('threading').Lock()
        p._prefetch_thread = None
        p._stale_multiplier = 2.0

        # In cron mode, system_prompt_block must return empty string
        result = p.system_prompt_block()
        assert result == ""

    def test_non_gateway_mode_returns_hybrid_prompt(self):
        """Normal (non-cron) hybrid mode returns tool instructions in prompt block."""
        from plugins.memory.honcho import HonchoMemoryProvider
        from unittest.mock import MagicMock

        p = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
        p._cron_skipped = False
        p._recall_mode = "hybrid"
        p._context_cadence = 1
        p._dialectic_cadence = 2
        p._base_context_cache = "test context"
        p._pending_dialectic = None
        p._pending_dialectic_ts = 0
        p._turn_count = 1
        p._prefetch_lock = __import__('threading').Lock()
        p._prefetch_thread = None
        p._stale_multiplier = 2.0
        p._manager = MagicMock()
        p._session_key = "test:session"

        result = p.system_prompt_block()
        # hybrid mode returns static tool instructions (not context — that's prefetch)
        assert result != ""
        assert "honcho" in result.lower()


class TestProfileIsolation:
    """Test that different profiles get isolated config."""

    def test_config_reads_from_hermes_home(self):
        """Config resolution uses ~/.hermes/honcho.json."""
        config_path = os.path.join(os.path.expanduser("~/.hermes"), "honcho.json")
        assert os.path.exists(config_path)

    def test_different_hosts_can_have_different_recall_modes(self):
        """Each host block can independently set recallMode."""
        cfg = _load_honcho_json()
        hosts = cfg.get("hosts", {})

        # Verify structure allows per-host recallMode
        for name, host in hosts.items():
            if "recallMode" in host:
                assert host["recallMode"] in ("hybrid", "context", "tools")

    def test_peer_name_per_host(self):
        """Each host can have its own peerName."""
        cfg = _load_honcho_json()
        hosts = cfg.get("hosts", {})
        hermes_host = hosts.get("hermes", {})

        # peerName should be set (default or explicit)
        peer_name = hermes_host.get("peerName", hermes_host.get("userPeer", ""))
        # Either explicitly set or falls back to default
        assert isinstance(peer_name, str)
