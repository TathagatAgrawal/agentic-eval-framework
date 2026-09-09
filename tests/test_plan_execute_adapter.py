"""Tests for PlanExecuteConfig, PlanExecuteAdapter, and its build_adapter wiring.

The LLM is mocked throughout (see tests/fakes.py) so this suite never calls the
real Gemini API.
"""

from decimal import Decimal

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.adapter import build_adapter
from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.contextualize import ContextualizeResult
from finance_qna.agent.plan_execute_adapter import PlanExecuteAdapter, PlanExecuteConfig
from finance_qna.agent.route import RouteDecision
from finance_qna.config import Settings
from tests.fakes import FakeLLM


def test_plan_execute_config_defaults() -> None:
    """PlanExecuteConfig must load with sensible defaults with no env overrides."""
    config = PlanExecuteConfig(_env_file=None)
    assert config.model == "gemini-3.5-flash-lite"
    assert config.groundedness_retry_limit == 1


def test_build_adapter_builds_a_plan_execute_adapter() -> None:
    """build_adapter must construct a PlanExecuteAdapter when configured to."""
    settings = Settings(google_api_key="fake-key", agent_architecture="plan_execute")  # type: ignore[call-arg]
    adapter = build_adapter(settings)
    assert isinstance(adapter, PlanExecuteAdapter)
    assert adapter.id == "plan_execute-gemini-3.5-flash-lite"


def _dining_tool_call(call_id: str = "call_1") -> dict:
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


def _fake_llm(
    question: str, plan_responses: list[AIMessage], answers: list[StructuredAnswer]
) -> FakeLLM:
    """Build a FakeLLM scripted to resolve `question` unambiguously, route it to
    'answer', serve `plan_responses` for successive plan/replan calls, and
    serve `answers` for successive draft calls."""
    return FakeLLM(
        structured_responses={
            "ContextualizeResult": [
                ContextualizeResult(resolved_question=question, is_ambiguous=False)
            ],
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": answers,
        },
        tool_call_responses=plan_responses,
    )


def test_run_turn_executes_every_planned_call_from_one_response(fixture_engine: Engine) -> None:
    """A single plan call's tool_calls must all be executed -- no interleaving,
    unlike the ReAct-based architectures."""
    question = "How much did I spend on dining in July 2024?"
    answer = StructuredAnswer(
        text="You spent $415.77 on dining in July 2024.",
        claims=[Claim(value=Decimal("415.77"), ledger_id="L1")],
    )
    fake_llm = _fake_llm(
        question, [AIMessage(content="", tool_calls=[_dining_tool_call()])], [answer]
    )

    adapter = PlanExecuteAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn(question, prior_turns=[])

    assert trace.route == "answer"
    assert len(trace.ledger) == 1
    assert trace.ledger[0]["tool_name"] == "aggregate_spending_tool"
    assert trace.groundedness_ok is True
    assert trace.retries == 0
    assert trace.final_answer == answer.text


def test_run_turn_replans_once_then_succeeds(fixture_engine: Engine) -> None:
    """An ungrounded first draft (from an empty first plan) must trigger exactly
    one re-plan, then accept a grounded redraft."""
    question = "How much did I spend on dining in July 2024?"
    ungrounded_answer = StructuredAnswer(
        text="You spent $999999.99 on dining.",
        claims=[Claim(value=Decimal("999999.99"), ledger_id="L1")],
    )
    grounded_answer = StructuredAnswer(
        text="You spent $415.77 on dining in July 2024.",
        claims=[Claim(value=Decimal("415.77"), ledger_id="L1")],
    )
    fake_llm = _fake_llm(
        question,
        [
            AIMessage(content="", tool_calls=[]),  # empty first plan -- nothing to cite
            AIMessage(content="", tool_calls=[_dining_tool_call("call_2")]),
        ],
        [ungrounded_answer, grounded_answer],
    )

    adapter = PlanExecuteAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn(question, prior_turns=[])

    assert trace.retries == 1
    assert trace.groundedness_ok is True
    assert trace.final_answer == grounded_answer.text
    assert len(trace.ledger) == 1  # the replan's one call, appended after the empty first plan


def test_run_turn_caveats_after_exhausting_retries(fixture_engine: Engine) -> None:
    """If every re-plan still produces an ungrounded draft, the turn must fall
    back to a caveated response instead of silently returning an unverified number."""
    question = "How much did I spend on dining in July 2024?"
    ungrounded_answer = StructuredAnswer(
        text="You spent $999999.99 on dining.",
        claims=[Claim(value=Decimal("999999.99"), ledger_id="L1")],
    )
    fake_llm = _fake_llm(
        question,
        [AIMessage(content="", tool_calls=[]), AIMessage(content="", tool_calls=[])],
        [ungrounded_answer, ungrounded_answer],
    )

    adapter = PlanExecuteAdapter(
        engine=fixture_engine,
        llm=fake_llm,  # type: ignore[arg-type]
        model_name="fake-model",
        groundedness_retry_limit=1,
    )
    trace = adapter.run_turn(question, prior_turns=[])

    assert trace.retries == 1
    assert trace.groundedness_ok is False
    assert "could not fully verify" in (trace.final_answer or "")


def test_run_turn_skips_an_unknown_planned_tool_without_raising(fixture_engine: Engine) -> None:
    """A hallucinated tool name in the plan response must be skipped, not fatal."""
    question = "How much did I spend on dining in July 2024?"
    plan_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "not_a_real_tool", "args": {}, "id": "call_bad"},
            _dining_tool_call("call_1"),
        ],
    )
    answer = StructuredAnswer(
        text="You spent $415.77 on dining in July 2024.",
        claims=[Claim(value=Decimal("415.77"), ledger_id="L1")],
    )
    fake_llm = _fake_llm(question, [plan_response], [answer])

    adapter = PlanExecuteAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn(question, prior_turns=[])

    assert len(trace.ledger) == 1
    assert trace.ledger[0]["ledger_id"] == "L1"


def test_run_turn_clarify_route_makes_no_plan_call(fixture_engine: Engine) -> None:
    """A 'clarify' decision must never reach the planning stage."""
    fake_llm = FakeLLM(
        structured_responses={
            "ContextualizeResult": [
                ContextualizeResult(
                    resolved_question="How much did I spend this month?",
                    is_ambiguous=True,
                    ambiguity_reason="no established time period",
                )
            ]
        },
        invoke_responses=[AIMessage(content="Which month did you mean?")],
    )

    adapter = PlanExecuteAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn("How much did I spend this month?", prior_turns=[])

    assert trace.route == "clarify"
    assert trace.ledger == []
    assert trace.final_answer == "Which month did you mean?"
