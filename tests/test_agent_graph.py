"""End-to-end tests of the compiled agent graph, with the LLM fully mocked.

These verify the graph wiring itself (act -> tool_node -> act -> draft_answer,
and the max_tool_steps loop guard) without spending any real Gemini quota. Tool
execution against the fixture database is real.
"""

from decimal import Decimal

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.graph import build_graph
from finance_qna.agent.state import initial_state
from tests.fakes import FakeAgentLLM


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


def test_graph_calls_one_tool_then_drafts_answer(fixture_engine: Engine) -> None:
    """The graph must run act -> tool_node -> act -> draft_answer for a simple question."""
    scripted_answer = StructuredAnswer(
        text="You spent $415.77 on dining in July 2024.",
        claims=[Claim(value=Decimal("415.77"), ledger_id="L1")],
    )
    fake_llm = FakeAgentLLM(
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call("call_1")]),
            AIMessage(content="I have enough information now."),
        ],
        final_answer=scripted_answer,
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=6)
    result = graph.invoke(initial_state("How much did I spend on dining in July 2024?"))

    assert len(result["ledger"]) == 1
    assert result["ledger"][0]["tool_name"] == "aggregate_spending_tool"
    assert result["draft_answer"] is scripted_answer


def test_graph_stops_at_max_tool_steps_even_if_model_keeps_requesting_tools(
    fixture_engine: Engine,
) -> None:
    """The loop guard must force draft_answer once max_tool_steps ledger entries exist,
    even when the model would otherwise keep requesting more tool calls."""
    scripted_answer = StructuredAnswer(text="Here's what I found so far.", claims=[])
    # Script more tool-call responses than max_tool_steps allows; the router should
    # never let act run enough times to exhaust this list.
    fake_llm = FakeAgentLLM(
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call(f"call_{i}")]) for i in range(10)
        ],
        final_answer=scripted_answer,
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=2)
    result = graph.invoke(initial_state("How much did I spend on dining?"))

    assert len(result["ledger"]) == 2
    assert result["draft_answer"] is scripted_answer
