"""Crash-consistency contracts for atomic persistence (T13-F10).

utils.atomic_write_text / atomic_json_write are the shared implementation for
EVERY destructive file rewrite in the codebase (memory store, skill manager,
curator state, config). The contracts pinned here:

* a serialization/encoding failure leaves the ORIGINAL content intact and NO
  temp-file debris in the directory (the BaseException cleanup path)
* KeyboardInterrupt mid-write is treated the same as a crash: original file
  intact, tmp cleaned, signal re-raised
* mode=0600 lands on the final file without a chmod-after-write window for
  NEW secret files (TOCTOU fix), and existing file modes are preserved when
  mode is not given
* parent directories are created and content round-trips byte-exact

The symlink-preservation surface is covered by test_atomic_replace_symlinks.py;
this file targets the failure/cleanup surface it does not touch.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from utils import atomic_json_write, atomic_write_text


def _tmp_files(path: Path) -> list[str]:
    return [p.name for p in path.parent.iterdir() if p.suffix == ".tmp"]


class TestAtomicWriteText:
    def test_roundtrip_and_parent_creation(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "file.txt"
        atomic_write_text(target, "héllo wörld\n")
        assert target.read_text(encoding="utf-8") == "héllo wörld\n"

    def test_encoding_failure_preserves_original_and_cleans_tmp(self, tmp_path):
        target = tmp_path / "data.txt"
        target.write_text("ORIGINAL", encoding="utf-8")
        with pytest.raises(UnicodeEncodeError):
            atomic_write_text(target, "ünïcode", encoding="ascii")
        assert target.read_text(encoding="utf-8") == "ORIGINAL"
        assert _tmp_files(tmp_path) == []

    def test_overwrite_replaces_content_completely(self, tmp_path):
        target = tmp_path / "f.txt"
        atomic_write_text(target, "a" * 10_000)
        atomic_write_text(target, "short")
        assert target.read_text(encoding="utf-8") == "short"


class TestAtomicJsonWriteFailure:
    def test_unserializable_data_preserves_original_and_cleans_tmp(self, tmp_path):
        target = tmp_path / "state.json"
        target.write_text('{"version": 1}', encoding="utf-8")
        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": object()})  # not JSON-serializable
        assert json.loads(target.read_text()) == {"version": 1}
        assert _tmp_files(tmp_path) == []

    def test_keyboard_interrupt_preserves_original_and_cleans_tmp(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "state.json"
        target.write_text('{"version": 1}', encoding="utf-8")

        def boom(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(json, "dump", boom)
        with pytest.raises(KeyboardInterrupt):
            atomic_json_write(target, {"x": 1})
        assert json.loads(target.read_text()) == {"version": 1}
        assert _tmp_files(tmp_path) == []

    def test_unserializable_data_on_new_file_leaves_nothing(self, tmp_path):
        target = tmp_path / "fresh.json"
        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": object()})
        assert not target.exists()
        assert _tmp_files(tmp_path) == []


class TestAtomicJsonWriteModes:
    def test_explicit_mode_applies_to_new_file(self, tmp_path):
        target = tmp_path / "secret.json"
        atomic_json_write(target, {"token": "x"}, mode=0o600)
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert json.loads(target.read_text()) == {"token": "x"}

    def test_existing_mode_preserved_when_mode_not_given(self, tmp_path):
        target = tmp_path / "state.json"
        target.write_text('{"a": 1}', encoding="utf-8")
        target.chmod(0o640)
        atomic_json_write(target, {"a": 2})
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        assert json.loads(target.read_text()) == {"a": 2}

    def test_default_new_file_mode_not_world_readable(self, tmp_path):
        # mkstemp creates 0o600; the restore path must not widen it.
        target = tmp_path / "creds.json"
        atomic_json_write(target, {"k": "v"})
        assert stat.S_IMODE(target.stat().st_mode) & 0o077 == 0
