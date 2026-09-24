"""Offline contracts for the MCP server. No browser, no paid APIs."""

import pytest

from laya_browser_ultrafast import mcp_server


def snapshot(status="ready", decision=None, history=()):
    return {
        "status": status,
        "goal": "Find a book",
        "page": {
            "url": "https://example.test/",
            "title": "Search",
            "text": "Search",
            "screenshot": "QUJD",
            "actions": [{"id": "e1", "node": 10, "rect": {}}],
        },
        "decision": decision,
        "history": list(history),
        "elapsed_ms": 10,
        "elements": [
            {"index": "1", "label": "Go", "role": "button", "operations": ["CLICK"], "value": ""},
        ],
    }


class FakeAgent:
    instances = []

    def __init__(self, url, goal):
        self.closed = False
        self.calls = []
        self.state = {"page": {"fingerprint": "fp"}}
        self._snapshot = snapshot()
        FakeAgent.instances.append(self)

    def snapshot(self):
        return self._snapshot

    def command(self, name, body=None):
        self.calls.append((name, body))
        if name == "predict":
            self._snapshot = snapshot(status="predicted", decision={
                "operation": "CLICK", "target": "1", "choice": "e1",
                "confidence": 0.9, "latency_ms": 5, "operation_probabilities": {"CLICK": 0.9},
            })
        elif name == "act":
            self._snapshot = snapshot(status="ready", history=[{
                "step": 1, "action": "Go", "operation": "CLICK",
                "text": None, "page_changed": True, "elapsed_ms": 20,
            }])
        elif name == "tick":
            self._snapshot = snapshot(status="done", history=[{
                "step": 1, "action": "Go", "operation": "CLICK",
                "text": None, "page_changed": True, "elapsed_ms": 20,
            }])
        return self._snapshot

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(mcp_server, "Agent", FakeAgent)
    monkeypatch.setattr(mcp_server, "AGENT", None, raising=False)
    FakeAgent.instances.clear()
    yield
    monkeypatch.setattr(mcp_server, "AGENT", None, raising=False)


def test_snapshot_without_task_errors():
    with pytest.raises(ValueError, match="start_task"):
        mcp_server.snapshot()


def test_start_task_rejects_empty_goal():
    with pytest.raises(ValueError, match="1-2,000"):
        mcp_server.start_task("https://example.test/", "  ")


def test_start_task_returns_element_table_without_screenshot():
    out = mcp_server.start_task("https://example.test/", "Find a book")
    assert out["status"] == "ready"
    assert out["elements"] == [
        {"index": "1", "label": "Go", "role": "button", "operations": ["CLICK"], "value": ""},
    ]
    assert "QUJD" not in str(out)


def test_predict_then_act_flow():
    mcp_server.start_task("https://example.test/", "Find a book")
    decision = mcp_server.predict()["decision"]
    assert decision["operation"] == "CLICK" and decision["target"] == "1"
    acted = mcp_server.act()
    assert acted["history"][0]["action"] == "Go"
    agent = FakeAgent.instances[-1]
    assert ("act", {"fingerprint": "fp"}) in agent.calls


def test_tick_and_run_to_done():
    mcp_server.start_task("https://example.test/", "Find a book")
    assert mcp_server.tick()["status"] == "done"
    out = mcp_server.run(max_ticks=5)
    assert out["status"] == "done"
    assert out["verify"]["url"] == "https://example.test/"


def test_run_respects_tick_budget():
    mcp_server.start_task("https://example.test/", "Find a book")
    agent = FakeAgent.instances[-1]
    agent.command = lambda name, body=None: snapshot(status="ready")
    out = mcp_server.run(max_ticks=3)
    assert out["status"] == "ready" and out["actions"] == 0


def test_close_task_closes_browser():
    mcp_server.start_task("https://example.test/", "Find a book")
    assert mcp_server.close_task() == {"status": "closed"}
    assert FakeAgent.instances[-1].closed
    with pytest.raises(ValueError, match="start_task"):
        mcp_server.snapshot()
