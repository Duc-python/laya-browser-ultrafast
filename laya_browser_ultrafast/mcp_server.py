"""MCP server for the Laya browser agent (stdio transport).

Tools wrap the same Agent loop as the demo inspector: start a task from one
natural-language goal, then predict/act/tick until done or blocked. A DONE
choice is not proof of success; verify the final page yourself.
"""

from mcp.server.fastmcp import FastMCP

from .agent import Agent
from .demo import load_environment

mcp = FastMCP("laya-browser-ultrafast")

AGENT = None


def _require_agent():
    if AGENT is None:
        raise ValueError("No task running. Call start_task(url, goal) first.")
    return AGENT


def _elements(snapshot):
    return [
        {
            "index": e.get("index"),
            "label": e.get("label"),
            "role": e.get("role"),
            "operations": e.get("operations", []),
            "value": e.get("value", ""),
        }
        for e in snapshot.get("elements", [])
    ]


def _decision(decision):
    if not decision:
        return None
    return {
        "operation": decision.get("operation"),
        "target": decision.get("target"),
        "choice": decision.get("choice"),
        "confidence": decision.get("confidence"),
        "latency_ms": decision.get("latency_ms"),
        "operation_probabilities": decision.get("operation_probabilities", {}),
    }


def _history(snapshot, limit=20):
    rows = []
    for h in snapshot.get("history", [])[-limit:]:
        rows.append({
            "step": h.get("step"),
            "action": h.get("action"),
            "operation": h.get("operation"),
            "text": h.get("text"),
            "page_changed": h.get("page_changed"),
            "elapsed_ms": h.get("elapsed_ms"),
        })
    return rows


def _summarize(snapshot):
    page = snapshot.get("page") or {}
    return {
        "status": snapshot.get("status"),
        "url": page.get("url"),
        "title": page.get("title"),
        "elapsed_ms": snapshot.get("elapsed_ms"),
        "actions": len(snapshot.get("history", [])),
        "elements": _elements(snapshot),
        "decision": _decision(snapshot.get("decision")),
        "history": _history(snapshot),
    }


@mcp.tool()
def start_task(url: str, goal: str) -> dict:
    """Open a fresh browser tab and start a task from one natural-language goal.

    Closes any previous task. Returns the first observed state: status, url,
    and the indexed element table to choose from.
    """
    global AGENT
    goal = (goal or "").strip()
    if not goal or len(goal) > 2000:
        raise ValueError("Enter a goal of 1-2,000 characters")
    if AGENT is not None:
        AGENT.close()
        AGENT = None
    AGENT = Agent(url, goal)
    return _summarize(AGENT.snapshot())


@mcp.tool()
def snapshot() -> dict:
    """Return the current state without deciding or acting: status, page,
    indexed elements, pending decision, and executed history."""
    return _summarize(_require_agent().snapshot())


@mcp.tool()
def predict() -> dict:
    """Ask the decision backend to choose one operation + target on the current
    page. Inspects the returned decision, then calls act (or tick to combine)."""
    agent = _require_agent()
    return {"decision": _decision(agent.command("predict", {})["decision"])}


@mcp.tool()
def act() -> dict:
    """Execute the pending decision from predict, then observe the result.
    A stale decision is consumed before any browser mutation; re-predict."""
    agent = _require_agent()
    page = agent.state["page"]
    return _summarize(agent.command("act", {"fingerprint": page["fingerprint"]}))


@mcp.tool()
def tick() -> dict:
    """Predict and execute in one step (re-observes on a stale page).
    Prefer tick for autonomous runs; use predict/act to inspect each choice."""
    return _summarize(_require_agent().command("tick", {}))


@mcp.tool()
def run(max_ticks: int = 30) -> dict:
    """Run ticks until done/blocked or max_ticks. Returns the final summary;
    verify the final page (url/title below) against the goal yourself."""
    agent = _require_agent()
    state = None
    for _ in range(max(1, max_ticks)):
        state = agent.command("tick", {})
        if state["status"] in {"done", "blocked"}:
            break
    summary = _summarize(state)
    page = (state.get("page") or {})
    summary["verify"] = {"url": page.get("url"), "title": page.get("title")}
    return summary


@mcp.tool()
def close_task() -> dict:
    """Close the browser tab and forget the current task."""
    global AGENT
    if AGENT is not None:
        AGENT.close()
        AGENT = None
    return {"status": "closed"}


def main():
    load_environment()
    mcp.run()


if __name__ == "__main__":
    main()
