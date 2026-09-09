"""Tests for compute_context_table, ContextStuffingConfig, and ContextStuffingAdapter.

The LLM is mocked throughout (see tests/fakes.py) so this suite never calls the
real Gemini API; compute_context_table itself is exercised against the real
fixture database since it's pure SQL-backed computation, no LLM involved.
"""

from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.adapter import build_adapter
from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.context_stuffing_adapter import (
    ContextStuffingAdapter,
    ContextStuffingConfig,
    compute_context_table,
)
from finance_qna.agent.contextualize import ContextualizeResult
from finance_qna.agent.route import RouteDecision
from finance_qna.config import Settings
from tests.fakes import FakeLLM


def test_context_stuffing_config_defaults() -> None:
    """ContextStuffingConfig must load with sensible defaults with no env overrides."""
    config = ContextStuffingConfig(_env_file=None)
    assert config.model == "gemini-3.5-flash-lite"


def test_build_adapter_builds_a_context_stuffing_adapter() -> None:
    """build_adapter must construct a ContextStuffingAdapter when configured to."""
    settings = Settings(google_api_key="fake-key", agent_architecture="context_stuffing")  # type: ignore[call-arg]
    adapter = build_adapter(settings)
    assert isinstance(adapter, ContextStuffingAdapter)
    assert adapter.id == "context_stuffing-gemini-3.5-flash-lite"


def test_compute_context_table_matches_ground_truth(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """The precomputed monthly totals must match the same ground truth the tool
    layer's own tests are checked against."""
    table = compute_context_table(fixture_engine)

    groceries_july_2024 = Decimal(
        ground_truth["monthly_category_totals"]["Groceries"]["2024-07-01"]
    )
    assert Decimal(table["monthly_category_totals"]["Groceries"]["2024-07"]) == groceries_july_2024


def test_compute_context_table_includes_the_labeled_subscription_change(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """The precomputed table must include the injected subscription price change."""
    table = compute_context_table(fixture_engine)
    event = ground_truth["subscription_change_event"]

    matching = [c for c in table["subscription_changes"] if c["merchant"] == event["merchant"]]
    assert len(matching) == 1
    assert matching[0]["new_amount"] == event["new_amount"]


def _fake_llm_for_answer(question: str, answer: StructuredAnswer) -> FakeLLM:
    """Build a FakeLLM scripted to resolve `question` unambiguously, route it to
    'answer', and produce `answer` as the draft (no tool_call_responses needed --
    this architecture makes no tool calls at all)."""
    return FakeLLM(
        structured_responses={
            "ContextualizeResult": [
                ContextualizeResult(resolved_question=question, is_ambiguous=False)
            ],
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": [answer],
        }
    )


def test_run_turn_answer_cites_the_synthetic_context_table_entry(fixture_engine: Engine) -> None:
    """An 'answer' turn must produce exactly one ledger entry: the synthetic
    context_table entry, never a real tool name."""
    question = "How much did I spend on groceries in July 2024?"
    answer = StructuredAnswer(
        text="You spent $511.94 on groceries in July 2024.",
        claims=[Claim(value=Decimal("511.94"), ledger_id="L1")],
    )
    fake_llm = _fake_llm_for_answer(question, answer)

    adapter = ContextStuffingAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn(question, prior_turns=[])

    assert trace.route == "answer"
    assert len(trace.ledger) == 1
    assert trace.ledger[0]["tool_name"] == "context_table"
    assert trace.ledger[0]["args"] == {}
    assert trace.groundedness_ok is True
    assert trace.retries == 0
    assert trace.final_answer == answer.text


def test_run_turn_caveats_an_ungrounded_draft_without_retrying(fixture_engine: Engine) -> None:
    """An ungrounded claim must produce a caveated answer, with no retry --
    the context doesn't change between attempts in this architecture."""
    question = "How much did I spend on groceries in July 2024?"
    ungrounded_answer = StructuredAnswer(
        text="You spent $999999.99 on groceries.",
        claims=[Claim(value=Decimal("999999.99"), ledger_id="L1")],
    )
    fake_llm = _fake_llm_for_answer(question, ungrounded_answer)

    adapter = ContextStuffingAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn(question, prior_turns=[])

    assert trace.groundedness_ok is False
    assert trace.retries == 0
    assert "could not fully verify" in (trace.final_answer or "")


def test_run_turn_clarify_route_makes_no_ledger_entry(fixture_engine: Engine) -> None:
    """A 'clarify' decision must produce an empty ledger -- the context table is
    only injected on the 'answer' path."""
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

    adapter = ContextStuffingAdapter(
        engine=fixture_engine,
        llm=fake_llm,
        model_name="fake-model",  # type: ignore[arg-type]
    )
    trace = adapter.run_turn("How much did I spend this month?", prior_turns=[])

    assert trace.route == "clarify"
    assert trace.ledger == []
    assert trace.groundedness_ok is True
    assert trace.final_answer == "Which month did you mean?"
