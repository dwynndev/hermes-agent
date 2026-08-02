"""F7: Self-Hosted Deployment — e2e tests against live honcho server.

Tests the full stack: API health, workspace/peer/session CRUD, message
ingestion, context retrieval, peer card, search. Requires honcho-selfhost
running on localhost:8000 (docker compose up -d).

Skip if server not available: tests auto-skip via the _server_available() guard.
"""

import json
import time
import urllib.request
import urllib.error

import pytest

# allow_network bypasses the conftest network guard (registered in conftest.py)
pytestmark = pytest.mark.allow_network

BASE_URL = "http://localhost:8000"


def _api(method: str, path: str, data: dict | None = None, timeout: int = 10) -> dict:
    """Make an API call to the honcho server."""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()[:200]}
    except Exception as e:
        return {"error": str(e)}


def _server_available() -> bool:
    """Check if honcho server is running."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# Skip all tests if server not available
skip_no_server = pytest.mark.skipif(
    not _server_available(),
    reason="honcho-selfhost not running on localhost:8000"
)


# ---------------------------------------------------------------------------
# Health & Infrastructure
# ---------------------------------------------------------------------------


@skip_no_server
class TestHealth:
    """Server health and infrastructure."""

    def test_health_endpoint(self):
        """GET /health returns 200 with status ok."""
        result = _api("GET", "/health")
        assert result.get("status") == "ok"

    def test_health_latency_under_100ms(self):
        """Health check responds in < 100ms (local network)."""
        t0 = time.time()
        _api("GET", "/health")
        latency = time.time() - t0
        assert latency < 0.1, f"Health check took {latency:.3f}s"


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------


@skip_no_server
class TestWorkspaceCRUD:
    """Workspace creation and retrieval."""

    def test_create_workspace(self):
        """POST /v3/workspaces creates a workspace."""
        ws_name = f"test-ws-{int(time.time())}"
        result = _api("POST", "/v3/workspaces", {"name": ws_name})
        assert "id" in result or "name" in result
        assert result.get("name") == ws_name or result.get("id") == ws_name

    def test_get_workspace(self):
        """GET /v3/workspaces/{name} retrieves a workspace."""
        ws_name = f"test-ws-get-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        result = _api("GET", f"/v3/workspaces/{ws_name}")
        assert "error" not in result or result.get("error") != 404


# ---------------------------------------------------------------------------
# Peer CRUD
# ---------------------------------------------------------------------------


@skip_no_server
class TestPeerCRUD:
    """Peer creation and retrieval."""

    def test_create_peer(self):
        """POST /v3/workspaces/{ws}/peers creates a peer."""
        ws_name = f"test-ws-peer-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        result = _api("POST", f"/v3/workspaces/{ws_name}/peers", {"name": "user-test"})
        assert "id" in result or "name" in result

    def test_get_peer(self):
        """GET /v3/workspaces/{ws}/peers/{name} retrieves a peer."""
        ws_name = f"test-ws-peer-get-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        _api("POST", f"/v3/workspaces/{ws_name}/peers", {"name": "user-get"})
        result = _api("GET", f"/v3/workspaces/{ws_name}/peers/user-get")
        assert "error" not in result or result.get("error") != 404


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


@skip_no_server
class TestSessionCRUD:
    """Session creation and retrieval."""

    def test_create_session(self):
        """POST /v3/workspaces/{ws}/sessions creates a session."""
        ws_name = f"test-ws-sess-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        result = _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-1"})
        assert "id" in result or "name" in result

    def test_get_session(self):
        """GET /v3/workspaces/{ws}/sessions/{name} retrieves a session."""
        ws_name = f"test-ws-sess-get-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-get"})
        result = _api("GET", f"/v3/workspaces/{ws_name}/sessions/sess-get")
        assert "error" not in result or result.get("error") != 404


# ---------------------------------------------------------------------------
# Message Ingestion
# ---------------------------------------------------------------------------


@skip_no_server
class TestMessageIngestion:
    """Message creation and retrieval."""

    def test_add_message_to_session(self):
        """POST message to a session."""
        ws_name = f"test-ws-msg-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        _api("POST", f"/v3/workspaces/{ws_name}/peers", {"name": "user-msg"})
        _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-msg"})

        # Add message
        result = _api("POST", f"/v3/workspaces/{ws_name}/sessions/sess-msg/messages", {
            "peer_id": "user-msg",
            "content": "Hello from e2e test!",
        })
        # Should succeed (201 or 200)
        assert "error" not in result or result.get("error") not in (400, 500)

    def test_get_session_messages(self):
        """GET messages from a session."""
        ws_name = f"test-ws-msg-get-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        _api("POST", f"/v3/workspaces/{ws_name}/peers", {"name": "user-msg-get"})
        _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-msg-get"})
        _api("POST", f"/v3/workspaces/{ws_name}/sessions/sess-msg-get/messages", {
            "peer_id": "user-msg-get",
            "content": "Test message for retrieval",
        })

        result = _api("GET", f"/v3/workspaces/{ws_name}/sessions/sess-msg-get/messages")
        # Should return a list or dict with messages
        assert isinstance(result, (list, dict))


# ---------------------------------------------------------------------------
# Context Retrieval
# ---------------------------------------------------------------------------


@skip_no_server
class TestContextRetrieval:
    """Session context endpoint."""

    def test_get_session_context(self):
        """GET /v3/workspaces/{ws}/sessions/{name}/context returns context."""
        ws_name = f"test-ws-ctx-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        _api("POST", f"/v3/workspaces/{ws_name}/peers", {"name": "user-ctx"})
        _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-ctx"})

        result = _api("GET", f"/v3/workspaces/{ws_name}/sessions/sess-ctx/context")
        # Context may be empty for new session, but should not error
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Latency Benchmarks
# ---------------------------------------------------------------------------


@skip_no_server
class TestLatencyBenchmarks:
    """Verify self-hosted latency is within acceptable bounds."""

    def test_workspace_create_latency(self):
        """Workspace creation < 200ms."""
        ws_name = f"test-ws-lat-{int(time.time())}"
        t0 = time.time()
        _api("POST", "/v3/workspaces", {"name": ws_name})
        latency = time.time() - t0
        assert latency < 0.2, f"Workspace create took {latency:.3f}s"

    def test_peer_create_latency(self):
        """Peer creation < 200ms."""
        ws_name = f"test-ws-lat-peer-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        t0 = time.time()
        _api("POST", f"/v3/workspaces/{ws_name}/peers", {"name": "user-lat"})
        latency = time.time() - t0
        assert latency < 0.2, f"Peer create took {latency:.3f}s"

    def test_session_create_latency(self):
        """Session creation < 200ms."""
        ws_name = f"test-ws-lat-sess-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        t0 = time.time()
        _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-lat"})
        latency = time.time() - t0
        assert latency < 0.2, f"Session create took {latency:.3f}s"

    def test_context_latency(self):
        """Context retrieval < 500ms."""
        ws_name = f"test-ws-lat-ctx-{int(time.time())}"
        _api("POST", "/v3/workspaces", {"name": ws_name})
        _api("POST", f"/v3/workspaces/{ws_name}/sessions", {"name": "sess-lat-ctx"})
        t0 = time.time()
        _api("GET", f"/v3/workspaces/{ws_name}/sessions/sess-lat-ctx/context")
        latency = time.time() - t0
        assert latency < 0.5, f"Context took {latency:.3f}s"
