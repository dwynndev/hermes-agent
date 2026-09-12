"""Wave 55 battery B01 race regression tests (W54-F001, W54-F011, W54-F012, W54-F013, W54-F014).

Every test starts a REAL race on the vulnerable code path and asserts the invariant the fix
guarantees. Synchronization is event/barrier-based only (zero sleeps):

- F001: two concurrent forks to the same explicit fork id -> exactly one 201, the loser 409.
- F011: a headerless platform retry (byte-identical payload) -> processed exactly once.
- F012: two DISTINCT headerless payloads in the same instant -> both processed (no false dup).
- F013: two threads racing MessageDeduplicator.is_duplicate on one id -> exactly one False.
- F014: a redelivered BlueBubbles webhook guid -> exactly one handle_message dispatch.
"""

import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import webhook as webhook_mod
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from gateway.platforms.helpers import MessageDeduplicator
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# F001 — concurrent fork sessions to the same explicit fork_id
# ---------------------------------------------------------------------------


class _FakeApiRequest:
    """Minimal aiohttp-request stand-in driving APIServerAdapter handlers directly."""

    def __init__(self, session_id: str, body: dict, headers: dict | None = None):
        self.match_info = {"session_id": session_id}
        self.headers = headers or {}
        self.query = {}
        self._body = json.dumps(body).encode("utf-8")

    async def json(self):
        return json.loads(self._body)

    async def read(self):
        return self._body


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def api_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.mark.asyncio
async def test_fork_session_concurrent_same_id_single_winner(api_adapter, session_db, monkeypatch):
    """Two simultaneous forks of the same source into the SAME fork id: the check-then-create
    window must not let both handlers pass the existence check (W54-F001)."""
    session_db.create_session("fork-src", "api_server", model="m-1", system_prompt="sys")
    session_db.replace_messages("fork-src", [{"role": "user", "content": "hi"}])
    fork_id = "fork-same-id"
    barrier = threading.Barrier(2)
    real_exec = session_db._execute_write

    def gated_exec(fn, *args, **kwargs):
        # Park both handlers at the atomic check-then-create BEFORE either enters the
        # write lock, so both observe the same pre-create state (W54-F001 race window).
        # Only the fork closure (defined in api_server.py) is gated; state-layer
        # writers (end_session/replace_messages) pass straight through.
        if getattr(fn, "__module__", "") == "gateway.platforms.api_server":
            barrier.wait(timeout=10)
        return real_exec(fn, *args, **kwargs)

    monkeypatch.setattr(session_db, "_execute_write", gated_exec)

    async def fork_once():
        resp = await api_adapter._handle_fork_session(_FakeApiRequest("fork-src", {"id": fork_id}))
        return resp.status

    statuses = await asyncio.gather(fork_once(), fork_once())
    assert sorted(statuses) == [201, 409]
    row = session_db.get_session(fork_id)
    assert row is not None and row.get("parent_session_id") == "fork-src"


# ---------------------------------------------------------------------------
# F011 / F012 — webhook fallback delivery-id stability and uniqueness
# ---------------------------------------------------------------------------


class _FakeWebhookRequest:
    def __init__(self, body: bytes, route: str = "ev"):
        self.match_info = {"route_name": route}
        self.headers = {}
        self.method = "POST"
        self._body = body
        self.content_length = len(body)

    async def read(self):
        return self._body


class _TickingClock:
    """Advancing clock: consecutive time.time() calls differ → distinct wall-clock fallbacks."""

    def __init__(self, start=1_700_000_000.0, step=1.5):
        self._t = start
        self._step = step

    def time(self):
        value = self._t
        self._t += self._step
        return value


class _FrozenClock:
    """Frozen clock: every time.time() call returns the same instant (same-millisecond window)."""

    def time(self):
        return 1_700_000_000.0


def _make_webhook_adapter() -> "webhook_mod.WebhookAdapter":
    return webhook_mod.WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1",
        "port": 0,
        "routes": {"ev": {"secret": webhook_mod._INSECURE_NO_AUTH, "prompt": "{text}"}},
    }))


@pytest.mark.asyncio
async def test_headerless_same_payload_retry_is_deduped(monkeypatch):
    """A platform retry (identical bytes, no delivery-id header) must not spawn a second run
    (W54-F011): the fallback key has to be stable across retries."""
    monkeypatch.setattr(webhook_mod, "time", _TickingClock())
    adapter = _make_webhook_adapter()
    adapter.handle_message = AsyncMock()
    body = json.dumps({"text": "release the build"}).encode("utf-8")

    first = await adapter._handle_webhook(_FakeWebhookRequest(body))
    second = await adapter._handle_webhook(_FakeWebhookRequest(body))
    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    assert first.status == 202
    assert second.status == 200  # duplicate — no second agent run


@pytest.mark.asyncio
async def test_headerless_distinct_payloads_same_instant_both_processed(monkeypatch):
    """Two DIFFERENT events landing in the same millisecond must not share a fallback id
    (W54-F012): the second would be silently dropped as a false duplicate."""
    monkeypatch.setattr(webhook_mod, "time", _FrozenClock())
    adapter = _make_webhook_adapter()
    adapter.handle_message = AsyncMock()

    for text in ("deploy red", "deploy blue"):
        body = json.dumps({"text": text}).encode("utf-8")
        resp = await adapter._handle_webhook(_FakeWebhookRequest(body))
        assert resp.status == 202  # both are fresh deliveries

    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    assert sorted(
        call.args[0].text for call in adapter.handle_message.await_args_list) == ["deploy blue", "deploy red"]


# ---------------------------------------------------------------------------
# F013 — MessageDeduplicator.is_duplicate must be atomic across threads
# ---------------------------------------------------------------------------


class _GatedSeen(dict):
    """Membership gate: the first checker computes its answer on the pre-store dict, then parks
    BEFORE returning, so the second checker is guaranteed to compute its answer before either
    thread records. The park is released either by the second checker's arrival (unfixed
    check-then-record) or by the test once it observes the first thread holding the dedupe lock
    (fixed code) — the lock then serializes the second check."""

    def __init__(self):
        super().__init__()
        self.first_parked = threading.Event()
        self.release = threading.Event()
        self._arrivals = 0
        self._arrivals_lock = threading.Lock()

    def __contains__(self, key):
        result = super().__contains__(key)
        with self._arrivals_lock:
            self._arrivals += 1
            arrivals = self._arrivals
            if arrivals == 1:
                self.first_parked.set()
            elif arrivals >= 2:
                self.release.set()
        if arrivals == 1:
            self.release.wait(timeout=15)
        return result


def test_deduplicator_check_and_record_is_atomic_across_threads():
    """Two threads racing is_duplicate on the same fresh id: exactly one must be told 'new'
    (W54-F013 — google_chat pubsub-thread vs loop-thread dedupe)."""
    dedup = MessageDeduplicator()
    gate = _GatedSeen()
    dedup._seen = gate
    results: list[bool] = []

    def worker():
        results.append(dedup.is_duplicate("same-guid"))

    first = threading.Thread(target=worker)
    first.start()
    assert gate.first_parked.wait(timeout=15)  # T1 computed membership, parked pre-record
    second = threading.Thread(target=worker)
    second.start()
    lock = getattr(dedup, "_lock", None)
    if lock is not None and lock.locked():
        # Fixed: T1 is parked INSIDE the dedupe lock — release it; the lock then serializes T2,
        # whose check must observe T1's record.
        gate.release.set()
    else:
        # Unfixed: the second checker's own arrival releases T1's parked store; both have
        # already computed "not seen" and both will record — the lost update under test.
        assert gate.release.wait(timeout=15)
    first.join(timeout=15)
    second.join(timeout=15)
    assert not first.is_alive() and not second.is_alive()
    assert sorted(results) == [False, True]


# ---------------------------------------------------------------------------
# F014 — BlueBubbles webhook redelivery dedupe
# ---------------------------------------------------------------------------


class _FakeBlueBubblesRequest:
    def __init__(self, payload: dict, password: str = "secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


def _make_bluebubbles_adapter() -> BlueBubblesAdapter:
    return BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234",
        "password": "secret",
        "send_read_receipts": False,
    }))


@pytest.mark.asyncio
async def test_webhook_redelivery_of_same_guid_dispatches_once(monkeypatch):
    """BlueBubbles redelivering a message (lost ack / server replay) must not spawn a second
    agent turn for the identical guid (W54-F014)."""
    adapter = _make_bluebubbles_adapter()
    handled: list[str] = []

    async def fake_handle_message(event):
        handled.append(event.message_id)

    monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
    payload = {
        "type": "new-message",
        "data": {
            "guid": "bb-guid-1",
            "text": "hello hermes",
            "handle": {"address": "+15555550100"},
            "isFromMe": False,
            "isGroup": False,
        },
    }
    await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
    await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    assert handled == ["bb-guid-1"]