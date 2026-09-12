"""W55 B07 (W54-F003): an IMMEDIATELY-acquired turn must still reload the
transcript when another writer's turn committed between this request's
pre-admission history read and its lease acquisition.

The pre-fix ``agent/turn_facade_lease.py`` reload only runs ``if waited:`` — an
acquire that succeeds on its FIRST attempt (no ``on_wait`` callback) keeps the
caller's stale history.  This test parks the reader on the acquire seam with
``threading.Event`` gates (zero sleeps) while the "writer" completes a full turn
(appends a message, bumps the live watermark), then lets the acquire succeed
immediately.  The admitted turn's view must contain the writer's message.

The captured watermark is passed via ``conversation_watermark=`` when the
signature supports it (post-fix).  Pre-fix the admission knows no watermark,
so the reload is skipped and the final assertion fails — an honest behavioural
red, not a missing-kwarg artefact.
"""
import inspect
import threading
from types import SimpleNamespace

from agent.turn_facade_lease import admit_durable_turn_lease

WRITER_MESSAGE = "WRITER-MESSAGE-W55-B07"


class _GatedDb:
    """Lease db mock whose acquire call is parked on an Event — the seam where
    the other writer's turn commits, exactly like the read->acquire window."""

    def __init__(self):
        self.acquire_started = threading.Event()
        self.writer_done = threading.Event()
        self.transcript = [{"role": "user", "content": "earlier user"},
                           {"role": "assistant", "content": "earlier assistant"}]
        self.watermark = len(self.transcript)  # MAX(active id) at history-read time
        self.on_wait_invocations = 0
        self.acquire_calls = 0
        self.reload_calls = 0

    def get_session(self, session_id):
        return {"id": session_id}

    def get_active_message_watermark(self, session_id):
        return self.watermark

    def resolve_resume_session_id(self, session_id):
        return session_id

    def get_messages_as_conversation(self, session_id, **kwargs):
        self.reload_calls += 1
        return list(self.transcript)

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.acquire_calls += 1
        raw_on_wait = kwargs.get("on_wait")

        def on_wait_probe(elapsed):
            # Only reachable if the production code decides a wait happened;
            # the fake never invokes it — an immediate acquire must not either.
            self.on_wait_invocations += 1
            if raw_on_wait is not None:
                raw_on_wait(elapsed)

        # Pre-admission pause: the writer's full turn lands while the reader is
        # parked here, BEFORE its first acquire attempt — so the attempt below
        # succeeds immediately and on_wait must never fire.
        self.acquire_started.set()
        assert self.writer_done.wait(timeout=30.0), "writer never signalled the seam"
        return True

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        pass


def _agent(db):
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        _persist_disabled=False,
        _interrupt_requested=False,
        _interrupt_message=None,
        _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0,
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    return agent


def _admit(agent, history):
    kwargs = {}
    if "conversation_watermark" in inspect.signature(admit_durable_turn_lease).parameters:
        kwargs["conversation_watermark"] = 2  # captured with the pre-admission read
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "cli"},
        conversation_history=history,
        **kwargs,
    )


def test_immediate_acquire_reloads_moved_transcript(monkeypatch):
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    db = _GatedDb()
    agent = _agent(db)
    stale = [{"role": "user", "content": "earlier user"}]  # pre-admission snapshot

    admission_box = {}

    def reader():
        admission_box["admission"] = _admit(agent, stale)

    thread = threading.Thread(target=reader, name="w55-b07-reader")
    thread.start()
    assert db.acquire_started.wait(timeout=30.0), "reader never reached the acquire seam"

    # Writer's full turn: transcript grows, live watermark moves past the
    # captured one.  The parked acquire then returns success on its first
    # attempt — an IMMEDIATE acquisition (waited=False).
    db.transcript.append({"role": "user", "content": WRITER_MESSAGE})
    db.watermark += 1
    db.writer_done.set()
    thread.join(timeout=30.0)
    assert not thread.is_alive(), "admission deadlocked on the writer gate"

    admission = admission_box["admission"]
    assert admission.early_result is None
    assert admission.lease is not None
    # The admission path must not have gone through the wait callback — this
    # exercises the immediate-acquire window, not the existing `waited` branch.
    assert db.acquire_calls == 1
    assert db.on_wait_invocations == 0

    # The bug contract: the admitted turn's view covers the writer's message.
    contents = [m.get("content") for m in admission.conversation_history]
    assert WRITER_MESSAGE in contents
    assert db.reload_calls >= 1

    admission.lease.release()