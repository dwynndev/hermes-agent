"""Negative-path tests for safety guardrails (T13-F5).

Pins the REFUSAL contracts — the cases where the agent MUST be stopped:

* file_safety._classify_write_denial / is_write_denied / get_write_denied_error:
  credential files+prefixes denied, state.db/sessions denied, HERMES_WRITE_SAFE_ROOT
  containment enforced, ordinary paths allowed.
* tool_guardrails.classify_tool_failure: terminal non-zero exit, memory-full,
  generic error markers, and the None-result passthrough.

A false ALLOW here is a security bug (secret overwrite, history falsification);
a false DENY breaks legitimate writes. Both directions are pinned.
"""

from __future__ import annotations

import os

from agent import file_safety as fs
from agent.tool_guardrails import classify_tool_failure


class TestWriteDenialCredentials:
    def test_exact_credential_paths_denied(self):
        # _classify_write_denial resolves the REAL home, so the contract test
        # must build the denied set against the real home too.
        home = os.path.realpath(os.path.expanduser("~"))
        denied = fs.build_write_denied_paths(home)
        for p in list(denied)[:5]:  # sample: set is large, 5 proves the loop
            assert fs._classify_write_denial(p) == "credential", p
            assert fs.is_write_denied(p) is True

    def test_ssh_key_write_denied(self):
        p = os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519")
        assert fs.is_write_denied(p) is True
        err = fs.get_write_denied_error(p)
        assert err is not None and "protected system/credential file" in err

    def test_credential_prefix_dirs_denied(self):
        home = os.path.realpath(os.path.expanduser("~"))
        for prefix in fs.build_write_denied_prefixes(home)[:3]:
            target = os.path.join(prefix, "nested", "file.txt")
            assert fs._classify_write_denial(target) == "credential", target


class TestWriteDenialSessionState:
    def test_state_db_write_denied(self):
        # Public contract: state.db is never writable via generic file tools
        # (falsifying conversation history). Internal classifier returns a
        # truthy value; the public API is what callers rely on.
        home_real = os.path.realpath(str(fs._hermes_home_path()))
        state_db = os.path.realpath(os.path.join(home_real, "state.db"))
        assert fs.is_write_denied(state_db) is True

    def test_sessions_dir_write_denied(self):
        home_real = os.path.realpath(str(fs._hermes_home_path()))
        inside = os.path.realpath(
            os.path.join(home_real, "sessions", "20260804_120000_abc", "transcript.jsonl")
        )
        assert fs.is_write_denied(inside) is True

    def test_mcp_tokens_and_pairing_denied_as_credential(self):
        home_real = os.path.realpath(str(fs._hermes_home_path()))
        for sub in ("mcp-tokens", "pairing"):
            p = os.path.realpath(os.path.join(home_real, sub, "creds.json"))
            assert fs._classify_write_denial(p) == "credential", sub


class TestSafeRoot:
    def test_safe_root_blocks_outside_paths(self, monkeypatch, tmp_path):
        allowed = tmp_path / "workspace"
        allowed.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(allowed))
        inside = str(allowed / "file.txt")
        outside = str(tmp_path / "elsewhere" / "file.txt")
        assert fs.is_write_denied(inside) is False
        assert fs._classify_write_denial(outside) == "safe_root"
        err = fs.get_write_denied_error(outside)
        assert err is not None and "HERMES_WRITE_SAFE_ROOT" in err

    def test_safe_root_prefix_siblings_not_confused(self, monkeypatch, tmp_path):
        # /safe/root must not allow /safe/root-evil (os.sep boundary)
        root = tmp_path / "root"
        root.mkdir()
        evil = tmp_path / "root-evil"
        evil.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(root))
        assert fs._classify_write_denial(str(evil / "x.txt")) == "safe_root"

    def test_no_safe_root_allows_ordinary_paths(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
        assert fs.is_write_denied(str(tmp_path / "normal.txt")) is False
        assert fs.get_write_denied_error(str(tmp_path / "normal.txt")) is None

    def test_multiple_safe_roots_split_by_pathsep(self, monkeypatch, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", f"{a}{os.pathsep}{b}")
        assert fs.is_write_denied(str(a / "f.txt")) is False
        assert fs.is_write_denied(str(b / "f.txt")) is False


class TestClassifyToolFailure:
    def test_none_result_is_not_failure(self):
        assert classify_tool_failure("terminal", None) == (False, "")

    def test_terminal_nonzero_exit_is_failure(self):
        ok, tag = classify_tool_failure("terminal", '{"exit_code": 2, "output": "x"}')
        assert ok is True and tag == " [exit 2]"

    def test_terminal_zero_exit_is_not_failure(self):
        assert classify_tool_failure("terminal", '{"exit_code": 0, "output": "ok"}') == (
            False, ""
        )

    def test_memory_full_is_classified(self):
        ok, tag = classify_tool_failure(
            "memory", '{"success": false, "error": "memory entries exceed the limit"}'
        )
        assert ok is True and tag == " [full]"

    def test_generic_error_marker_is_failure(self):
        ok, tag = classify_tool_failure("web_search", '{"error": "timeout"}')
        assert ok is True and tag == " [error]"
        ok2, _ = classify_tool_failure("anything", "Error: something exploded")
        assert ok2 is True

    def test_clean_result_not_failure(self):
        assert classify_tool_failure("web_search", '{"results": []}') == (False, "")
