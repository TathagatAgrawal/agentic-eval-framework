"""End-to-end tests of the compiled agent graph, with the LLM fully mocked.

These verify the graph wiring itself -- route -> act -> tool_node -> act ->
draft_answer -> ground_check -> respond (or retry / caveat), and the
clarify/refuse short-circuit paths -- without spending any real Gemini quota.
Tool execution against the fixture database is real, so groundedness checks in
these tests run against genuine tool results, not scripted numbers.
"""

from decimal import Decimal

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.graph import build_graph
from finance_qna.agent.route import RouteDecision
from finance_qna.agent.state import initial_state
from finance_qna.tools.models import DateRange, TransactionFilter
from finance_qna.tools.query_tools import aggregate_spending
from tests.fakes import FakeLLM


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


def _real_dining_july_2024_total(fixture_engine: Engine) -> Decimal:
    """Look up the real Dining/July-2024 total directly, so grounded test claims
    match what the tool actually returns rather than an arbitrary scripted value."""
    filt = TransactionFilter(
        category="Dining", date_range=DateRange(start="2024-07-01", end="2024-07-31")
    )
    result = aggregate_spending(fixture_engine, filt)
    assert not isinstance(result, list)
    return result.total


def test_graph_calls_one_tool_then_respond_with_a_grounded_answer(fixture_engine: Engine) -> None:
    """The graph must run route -> act -> tool_node -> act -> draft_answer ->
    ground_check -> respond when the draft answer is properly grounded."""
    real_total = _real_dining_july_2024_total(fixture_engine)
    scripted_answer = StructuredAnswer(
        text=f"You spent ${real_total} on dining in July 2024.",
        claims=[Claim(value=real_total, ledger_id="L1")],
    )
    fake_llm = FakeLLM(
        structured_responses={
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": [scripted_answer],
        },
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call("call_1")]),
            AIMessage(content="I have enough information now."),
        ],
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=6)
    result = graph.invoke(initial_state("How much did I spend on dining in July 2024?"))

    assert result["route"] == "answer"
    assert len(result["ledger"]) == 1
    assert result["groundedness_ok"] is True
    assert result["final_answer"] == scripted_answer.text


def test_graph_stops_at_max_tool_steps_even_if_model_keeps_requesting_tools(
    fixture_engine: Engine,
) -> None:
    """The loop guard must force draft_answer once max_tool_steps ledger entries exist,
    even when the model would otherwise keep requesting more tool calls."""
    scripted_answer = StructuredAnswer(text="Here's what I found so far.", claims=[])
    # Script more tool-call responses than max_tool_steps allows; the router should
    # never let act run enough times to exhaust this list.
    fake_llm = FakeLLM(
        structured_responses={
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": [scripted_answer],
        },
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call(f"call_{i}")]) for i in range(10)
        ],
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=2)
    result = graph.invoke(initial_state("How much did I spend on dining?"))

    assert len(result["ledger"]) == 2
    assert result["final_answer"] == scripted_answer.text


def test_graph_retries_once_then_succeeds_after_ungrounded_draft(fixture_engine: Engine) -> None:
    """An ungrounded first draft must trigger exactly one retry through act, then
    a corrected, grounded second draft must be accepted."""
    real_total = _real_dining_july_2024_total(fixture_engine)
    ungrounded_answer = StructuredAnswer(
        text="You spent $999999.99 on dining.",
        claims=[Claim(value=Decimal("999999.99"), ledger_id="L1")],
    )
    grounded_answer = StructuredAnswer(
        text=f"You spent ${real_total} on dining in July 2024.",
        claims=[Claim(value=real_total, ledger_id="L1")],
    )
    fake_llm = FakeLLM(
        structured_responses={
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": [ungrounded_answer, grounded_answer],
        },
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call("call_1")]),
            AIMessage(content="I have enough information now."),
            # after the retry note, act is called a third time; no more tool
            # calls are needed since L1 is already in the ledger
            AIMessage(content="I'll revise my answer using the existing data."),
        ],
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=6, groundedness_retry_limit=1)
    result = graph.invoke(initial_state("How much did I spend on dining in July 2024?"))

    assert result["retry_count"] == 1
    assert result["groundedness_ok"] is True
    assert result["final_answer"] == grounded_answer.text


def test_graph_falls_back_to_caveat_after_exhausting_retries(fixture_engine: Engine) -> None:
    """If every retry still produces an ungrounded draft, the graph must fall back
    to a caveated response instead of silently returning an unverified number."""
    ungrounded_answer = StructuredAnswer(
        text="You spent $999999.99 on dining.",
        claims=[Claim(value=Decimal("999999.99"), ledger_id="L1")],
    )
    fake_llm = FakeLLM(
        structured_responses={
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": [ungrounded_answer, ungrounded_answer],
        },
        tool_call_responses=[
            AIMessage(content="", tool_calls=[_tool_call("call_1")]),
            AIMessage(content="I have enough information now."),
        ],
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=6, groundedness_retry_limit=0)
    result = graph.invoke(initial_state("How much did I spend on dining in July 2024?"))

    assert result["groundedness_ok"] is False
    assert result["final_answer"] is not None
    assert "could not fully verify" in result["final_answer"]
    assert ungrounded_answer.text in result["final_answer"]


def test_graph_clarify_path_never_calls_any_tool(fixture_engine: Engine) -> None:
    """When route chooses "clarify", the graph must go straight to a clarifying
    question with zero tool calls and an empty ledger."""
    fake_llm = FakeLLM(
        structured_responses={
            "RouteDecision": [RouteDecision(route="clarify", reason="ambiguous time period")],
        },
        invoke_responses=[AIMessage(content="Which month did you mean?")],
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=6)
    result = graph.invoke(initial_state("How much did I spend this month?"))

    assert result["route"] == "clarify"
    assert result["ledger"] == []
    assert result["final_answer"] == "Which month did you mean?"


def test_graph_refuse_path_never_calls_any_tool(fixture_engine: Engine) -> None:
    """When route chooses "refuse", the graph must go straight to a decline
    message with zero tool calls and an empty ledger."""
    fake_llm = FakeLLM(
        structured_responses={
            "RouteDecision": [RouteDecision(route="refuse", reason="not about the data")],
        },
        invoke_responses=[AIMessage(content="I can only answer questions about your spending.")],
    )

    graph = build_graph(fixture_engine, fake_llm, max_tool_steps=6)
    result = graph.invoke(initial_state("Should I invest in index funds?"))

    assert result["route"] == "refuse"
    assert result["ledger"] == []
    assert result["final_answer"] == "I can only answer questions about your spending."
