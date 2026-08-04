"""Config round-trip + cache contracts (T13-F8).

The config path DEFAULT_CONFIG -> config.yaml -> load_config() is read on
EVERY agent turn. A round-trip or cache bug silently changes runtime
behavior. Contracts pinned here (each in a fresh subprocess with a temp
HERMES_HOME so the module-level _LOAD_CONFIG_CACHE cannot cross-pollinate):

* DEFAULT_CONFIG serializes to YAML and parses back to an identical structure
* load_config() returns a DEEPCOPY: mutating the result never corrupts the
  cached config seen by the next caller
* load_config_readonly() returns the SAME cached object (the documented
  fast path) until the file changes
* editing config.yaml invalidates the cache (mtime/size keyed) — the next
  read sees the new values without a process restart
* YAML 1.1 bool coercion: a bare `on` key in the file parses as boolean
  True under PyYAML — the loader must surface whatever the YAML layer
  produced (documents the trap; config consumers must not rely on string keys
  named on/off/yes/no staying strings)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_py(code: str, home: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestDefaultsRoundTrip:
    def test_default_config_survives_yaml_round_trip(self, tmp_path):
        out = _run_py(
            """
import json, yaml
from hermes_cli.config_defaults import DEFAULT_CONFIG
s = yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False)
back = yaml.safe_load(s)
print(json.dumps({"equal": back == DEFAULT_CONFIG}))
""",
            tmp_path,
        )
        assert out["equal"] is True


class TestCacheContracts:
    def test_load_config_deepcopy_does_not_corrupt_cache(self, tmp_path):
        (tmp_path / "config.yaml").write_text("agent:\n  max_turns: 42\n")
        out = _run_py(
            """
import json
from hermes_cli.config import load_config
c1 = load_config()
c1["agent"]["max_turns"] = 999   # mutate the returned copy
c2 = load_config()                 # cache hit must be pristine
print(json.dumps({"second": c2["agent"]["max_turns"]}))
""",
            tmp_path,
        )
        assert out["second"] == 42

    def test_readonly_returns_cached_object(self, tmp_path):
        (tmp_path / "config.yaml").write_text("agent:\n  max_turns: 42\n")
        out = _run_py(
            """
import json
from hermes_cli.config import load_config_readonly
c1 = load_config_readonly()
c2 = load_config_readonly()
print(json.dumps({"same_object": c1 is c2}))
""",
            tmp_path,
        )
        assert out["same_object"] is True

    def test_file_edit_invalidates_cache(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("agent:\n  max_turns: 42\n")
        out = _run_py(
            """
import json, os, time
from hermes_cli.config import load_config
before = load_config()["agent"]["max_turns"]
path = os.path.join(os.environ["HERMES_HOME"], "config.yaml")
# bump mtime_ns AND size so the (mtime_ns, size) cache key changes
with open(path, "w") as f:
    f.write("agent:\\n  max_turns: 77\\n")
os.utime(path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
after = load_config()["agent"]["max_turns"]
print(json.dumps({"before": before, "after": after}))
""",
            tmp_path,
        )
        assert out["before"] == 42 and out["after"] == 77


class TestYamlBoolCoercionTrap:
    def test_bare_on_key_becomes_boolean(self, tmp_path):
        # Documents the YAML 1.1 trap: unquoted `on` is parsed as True.
        # This test PINS the trap so any future loader change that starts
        # silently swallowing such keys is caught.
        (tmp_path / "config.yaml").write_text("hooks:\n  on: something\n")
        out = _run_py(
            """
import json
from hermes_cli.config import load_config_readonly
cfg = load_config_readonly()
hooks = cfg.get("hooks", {})
print(json.dumps({"has_bool_true": True in hooks, "keys": [str(k) for k in hooks]}))
""",
            tmp_path,
        )
        assert out["has_bool_true"] is True
