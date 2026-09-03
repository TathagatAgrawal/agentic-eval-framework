"""Unit tests for the individual agent node functions, with the LLM mocked out.

These exercise real tool execution against the fixture database but never call
the real Gemini API, per the project's directive to keep test-suite runs free.
"""

import json
from decimal import Decimal

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.nodes import (
    make_act_node,
    make_draft_answer_node,
    make_router,
    make_tool_node,
)
from finance_qna.agent.state import AgentState, initial_state
from finance_qna.tools.registry import build_tools
from tests.fakes import FakeStructuredLLM, FakeToolCallingLLM


def _aggregate_tool_call(call_id: str = "call_1") -> dict:
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


def test_tool_node_appends_ledger_entry_on_success(fixture_engine: Engine) -> None:
    """A successful tool call must produce a ledger entry and a matching ToolMessage."""
    tools = build_tools(fixture_engine)
    tool_node = make_tool_node(tools)

    ai_message = AIMessage(content="", tool_calls=[_aggregate_tool_call()])
    state = initial_state("How much did I spend on dining in July 2024?")
    state["messages"] = [*state["messages"], ai_message]

    update = tool_node(state)

    assert len(update["ledger"]) == 1
    entry = update["ledger"][0]
    assert entry["ledger_id"] == "L1"
    assert entry["tool_name"] == "aggregate_spending_tool"
    assert Decimal(entry["result"]["total"]) > 0

    tool_message = update["messages"][-1]
    payload = json.loads(tool_message.content)
    assert payload["ledger_id"] == "L1"


def test_tool_node_reports_unknown_category_without_ledger_entry(fixture_engine: Engine) -> None:
    """An unknown category must produce an error message and no ledger entry."""
    tools = build_tools(fixture_engine)
    tool_node = make_tool_node(tools)

    bad_call = _aggregate_tool_call()
    bad_call["args"]["filt"]["category"] = "Not A Real Category"
    ai_message = AIMessage(content="", tool_calls=[bad_call])
    state = initial_state("How much did I spend on fake stuff?")
    state["messages"] = [*state["messages"], ai_message]

    update = tool_node(state)

    assert update["ledger"] == []
    payload = json.loads(update["messages"][-1].content)
    assert "error" in payload
    assert "Dining" in payload["valid_categories"]


def test_router_continues_to_tool_node_when_tool_calls_pending() -> None:
    """The router must send the graph to tool_node when the last AI message has tool calls."""
    router = make_router(max_tool_steps=6)
    state = initial_state("q")
    state["messages"] = [
        *state["messages"],
        AIMessage(content="", tool_calls=[_aggregate_tool_call()]),
    ]

    assert router(state) == "tool_node"


def test_router_stops_when_ledger_hits_max_tool_steps() -> None:
    """The router must force draft_answer once the ledger reaches max_tool_steps,
    even if the model is still requesting more tool calls."""
    router = make_router(max_tool_steps=1)
    state = initial_state("q")
    state["messages"] = [
        *state["messages"],
        AIMessage(content="", tool_calls=[_aggregate_tool_call()]),
    ]
    state["ledger"] = [
        {
            "ledger_id": "L1",
            "tool_name": "aggregate_spending_tool",
            "args": {},
            "result": {},
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
    ]

    assert router(state) == "draft_answer"


def test_router_stops_when_no_tool_calls_requested() -> None:
    """The router must go to draft_answer when the model stops requesting tools."""
    router = make_router(max_tool_steps=6)
    state = initial_state("q")
    state["messages"] = [*state["messages"], AIMessage(content="I'm ready to answer.")]

    assert router(state) == "draft_answer"


def test_act_node_appends_llm_response_to_messages() -> None:
    """The act node must append whatever the LLM returns to the message history."""
    scripted_response = AIMessage(content="", tool_calls=[_aggregate_tool_call()])
    fake_llm = FakeToolCallingLLM([scripted_response])
    act = make_act_node(fake_llm, tools=[])

    state = initial_state("How much did I spend on dining in July 2024?")
    update = act(state)

    assert update["messages"][-1] is scripted_response
    assert len(update["messages"]) == len(state["messages"]) + 1


def test_draft_answer_node_returns_scripted_structured_answer() -> None:
    """The draft_answer node must call the structured-output LLM and store its result."""
    scripted_answer = StructuredAnswer(
        text="You spent $415.77 on dining.",
        claims=[Claim(value=Decimal("415.77"), ledger_id="L1")],
    )
    fake_llm = FakeStructuredLLM(scripted_answer)
    draft_answer = make_draft_answer_node(fake_llm)

    state: AgentState = initial_state("How much did I spend on dining?")
    state["ledger"] = [
        {
            "ledger_id": "L1",
            "tool_name": "aggregate_spending_tool",
            "args": {},
            "result": {"total": "415.77", "count": 8},
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
    ]

    update = draft_answer(state)

    assert update["draft_answer"] is scripted_answer
