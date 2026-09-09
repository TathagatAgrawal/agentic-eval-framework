"""Tests for MonolithicReactConfig, MonolithicReactAdapter, and its build_adapter wiring.

The LLM is mocked throughout (see tests/fakes.py) so this suite never calls the
real Gemini API.
"""

from decimal import Decimal

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.adapter import build_adapter
from finance_qna.agent.answer import Claim
from finance_qna.agent.monolithic_adapter import (
    MonolithicReactAdapter,
    MonolithicReactConfig,
    MonolithicResult,
)
from finance_qna.config import Settings
from tests.fakes import FakeLLM


def test_monolithic_config_defaults() -> None:
    """MonolithicReactConfig must load with sensible defaults with no env overrides."""
    config = MonolithicReactConfig(_env_file=None)
    assert config.model == "gemini-3.5-flash-lite"
    assert config.max_tool_steps == 6


def test_build_adapter_builds_a_monolithic_adapter() -> None:
    """build_adapter must construct a MonolithicReactAdapter when configured to."""
    settings = Settings(google_api_key="fake-key", agent_architecture="monolithic")  # type: ignore[call-arg]
    adapter = build_adapter(settings)
    assert isinstance(adapter, MonolithicReactAdapter)
    assert adapter.id == "monolithic-gemini-3.5-flash-lite"


def _tool_call(call_id: str) -> dict:
    """Build a tool_call dict requesting Dining spend for July 2024."""
    return {
        "name": "aggregate_spending_tool",
        "args": {
            "filt": {
                "category": "Dining",
                "date_range": {"start": "2024-07-01", "end": "2024-07-31"},
            },
            "group_by": "none",
        },
        "id": call_id,
    }


def test_run_turn_calls_tools_then_produces_a_grounded_answer(fixture_engine: Engine) -> None:
    """A tool call followed by a grounded MonolithicResult must produce a
    matching RunTrace with the real ledger and no retries."""
    final_result = MonolithicResult(
        route="answer",
        text="You spent $415.77 on dining in July 2024.",
        claims=[Claim(value=Decimal("415.77"), ledger_id="L1")],
    )
    fake_llm = FakeLLM(
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call("call_1")]),
            AIMessage(content="I have enough information now."),
        ],
        structured_responses={"MonolithicResult": [final_result]},
    )

    adapter = MonolithicReactAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn("How much did I spend on dining in July 2024?", prior_turns=[])

    assert trace.route == "answer"
    assert len(trace.ledger) == 1
    assert trace.ledger[0]["tool_name"] == "aggregate_spending_tool"
    assert trace.groundedness_ok is True
    assert trace.retries == 0
    assert trace.final_answer == final_result.text
    assert trace.resolved_question == "How much did I spend on dining in July 2024?"


def test_run_turn_caveats_an_ungrounded_draft_without_retrying(fixture_engine: Engine) -> None:
    """An ungrounded claim must produce a caveated answer, with no retry loop --
    this architecture makes exactly one final-decision call per turn."""
    ungrounded_result = MonolithicResult(
        route="answer",
        text="You spent $999999.99 on dining.",
        claims=[Claim(value=Decimal("999999.99"), ledger_id="L1")],
    )
    fake_llm = FakeLLM(
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call("call_1")]),
            AIMessage(content="I have enough information now."),
        ],
        structured_responses={"MonolithicResult": [ungrounded_result]},
    )

    adapter = MonolithicReactAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn("How much did I spend on dining in July 2024?", prior_turns=[])

    assert trace.groundedness_ok is False
    assert trace.retries == 0
    assert "could not fully verify" in (trace.final_answer or "")
    assert ungrounded_result.text in (trace.final_answer or "")


def test_run_turn_clarify_route_makes_no_tool_calls(fixture_engine: Engine) -> None:
    """A 'clarify' decision must never have required a tool call, and must be
    trivially grounded (no claims to check)."""
    clarify_result = MonolithicResult(route="clarify", text="Which month did you mean?", claims=[])
    fake_llm = FakeLLM(
        tool_call_responses=[AIMessage(content="I'll decide how to respond now.")],
        structured_responses={"MonolithicResult": [clarify_result]},
    )

    adapter = MonolithicReactAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn("How much did I spend this month?", prior_turns=[])

    assert trace.route == "clarify"
    assert trace.ledger == []
    assert trace.groundedness_ok is True
    assert trace.final_answer == "Which month did you mean?"


def test_run_turn_respects_max_tool_steps(fixture_engine: Engine) -> None:
    """The tool loop must stop once max_tool_steps ledger entries exist, even if
    the model keeps requesting more tool calls."""
    final_result = MonolithicResult(route="answer", text="Here's what I found.", claims=[])
    fake_llm = FakeLLM(
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call(f"call_{i}")]) for i in range(10)
        ],
        structured_responses={"MonolithicResult": [final_result]},
    )

    adapter = MonolithicReactAdapter(
        engine=fixture_engine,
        llm=fake_llm,  # type: ignore[arg-type]
        model_name="fake-model",
        max_tool_steps=2,
    )
    trace = adapter.run_turn("How much did I spend on dining?", prior_turns=[])

    assert len(trace.ledger) == 2
