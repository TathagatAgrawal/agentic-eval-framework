"""Tests for building and persisting per-turn run traces."""

from decimal import Decimal
from pathlib import Path

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.state import initial_state
from finance_qna.tracing.trace import build_trace, read_trace, write_trace


def _completed_state():
    """Build a state as it would look after a full, successful graph run."""
    state = initial_state("How much did I spend on dining in July 2024?")
    state["resolved_question"] = "How much did I spend on dining in July 2024?"
    state["route"] = "answer"
    state["ledger"] = [
        {
            "ledger_id": "L1",
            "tool_name": "aggregate_spending_tool",
            "args": {},
            "result": {"total": "415.77", "count": 8},
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
    ]
    state["draft_answer"] = StructuredAnswer(
        text="You spent $415.77 on dining.", claims=[Claim(value=Decimal("415.77"), ledger_id="L1")]
    )
    state["groundedness_ok"] = True
    state["retry_count"] = 0
    state["final_answer"] = "You spent $415.77 on dining."
    return state


def test_build_trace_captures_the_full_turn() -> None:
    """build_trace must copy every relevant field off the completed state."""
    state = _completed_state()

    trace = build_trace(state, turn_id="1", latency_ms=1234)

    assert trace.turn_id == "1"
    assert trace.question == state["question"]
    assert trace.resolved_question == state["resolved_question"]
    assert trace.route == "answer"
    assert trace.ledger == state["ledger"]
    assert trace.groundedness_ok is True
    assert trace.retries == 0
    assert trace.final_answer == state["final_answer"]
    assert trace.latency_ms == 1234


def test_write_trace_then_read_trace_round_trips(tmp_path: Path) -> None:
    """A written trace must read back byte-for-byte equivalent."""
    state = _completed_state()
    trace = build_trace(state, turn_id="1", latency_ms=42)

    path = write_trace(trace, tmp_path / "session-abc")

    assert path == tmp_path / "session-abc" / "1.json"
    assert path.exists()
    assert read_trace(path) == trace
