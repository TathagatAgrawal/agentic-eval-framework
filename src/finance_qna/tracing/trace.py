"""Per-turn run traces: the "show its reasoning transparently" feature.

Built from a completed `AgentState` by the caller (the CLI, or later the eval
runner) rather than written from inside a graph node -- consistent with how
`TurnMemory` is built via `state.turn_memory_from_state` instead of a node
mutating an external object. This keeps the graph a pure function of state, and
is also the eval harness's sole planned way to inspect what a run actually did:
question -> resolved question -> ledger -> groundedness verdict -> final answer.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.state import AgentState, LedgerEntry


class RunTrace(BaseModel):
    """A complete, JSON-serializable record of one turn's reasoning and result."""

    turn_id: str
    question: str
    resolved_question: str | None
    route: str | None
    ledger: list[LedgerEntry]
    draft_answer: StructuredAnswer | None
    groundedness_ok: bool
    retries: int
    final_answer: str | None
    latency_ms: int


def build_trace(state: AgentState, turn_id: str, latency_ms: int) -> RunTrace:
    """Construct a `RunTrace` from a completed turn's final `AgentState`."""
    return RunTrace(
        turn_id=turn_id,
        question=state["question"],
        resolved_question=state["resolved_question"],
        route=state["route"],
        ledger=state["ledger"],
        draft_answer=state["draft_answer"],
        groundedness_ok=state["groundedness_ok"],
        retries=state["retry_count"],
        final_answer=state["final_answer"],
        latency_ms=latency_ms,
    )


def write_trace(trace: RunTrace, run_dir: Path) -> Path:
    """Write `trace` as JSON to `run_dir/<turn_id>.json`, creating `run_dir` if needed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{trace.turn_id}.json"
    path.write_text(trace.model_dump_json(indent=2))
    return path


def read_trace(path: Path) -> RunTrace:
    """Read a previously written trace back from disk."""
    return RunTrace.model_validate(json.loads(path.read_text()))
