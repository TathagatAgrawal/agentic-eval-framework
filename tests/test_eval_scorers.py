"""Unit tests for the six eval scorers, against hand-built RunTrace fixtures.

No agent, no adapter, no LLM anywhere -- these test the scoring logic alone,
per design/eval-harness-plan.md's build order step 3.
"""

from decimal import Decimal

from eval.schema import ExpectedResult
from eval.scorers import (
    clarification,
    contextual_correctness,
    efficiency,
    groundedness,
    numeric_correctness,
    refusal,
)
from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.tracing.trace import RunTrace


def _trace(
    route: str | None = "answer",
    ledger: list | None = None,
    claims: list[Claim] | None = None,
    groundedness_ok: bool = True,
    retries: int = 0,
    final_answer: str | None = "an answer",
) -> RunTrace:
    """Build a RunTrace fixture with sensible defaults, overridable per test."""
    return RunTrace(
        turn_id="1",
        question="a question",
        resolved_question="a question",
        route=route,
        ledger=ledger or [],
        draft_answer=StructuredAnswer(text=final_answer or "", claims=claims or []),
        groundedness_ok=groundedness_ok,
        retries=retries,
        final_answer=final_answer,
        latency_ms=100,
    )


def _ledger_entry(ledger_id: str, tool_name: str, args: dict, result: dict) -> dict:
    """Build a ledger entry dict matching the LedgerEntry TypedDict shape."""
    return {
        "ledger_id": ledger_id,
        "tool_name": tool_name,
        "args": args,
        "result": result,
        "timestamp": "2025-01-01T00:00:00+00:00",
    }


# --- numeric_correctness ---


def test_numeric_correctness_passes_when_claim_matches_within_tolerance() -> None:
    """A claim within tolerance of the expected value must pass."""
    trace = _trace(claims=[Claim(value=Decimal("511.94"), ledger_id="L1")])
    expected = ExpectedResult(behavior="answer", value=511.94, tolerance=0.01)
    result = numeric_correctness.score(trace, expected)
    assert result.passed is True


def test_numeric_correctness_fails_when_no_claim_matches() -> None:
    """A claim far from the expected value must fail."""
    trace = _trace(claims=[Claim(value=Decimal("100.00"), ledger_id="L1")])
    expected = ExpectedResult(behavior="answer", value=511.94, tolerance=0.01)
    result = numeric_correctness.score(trace, expected)
    assert result.passed is False


def test_numeric_correctness_handles_multiple_expected_values() -> None:
    """A comparison case with two expected values must pass when both are matched."""
    trace = _trace(
        claims=[
            Claim(value=Decimal("1325.97"), ledger_id="L1"),
            Claim(value=Decimal("1613.01"), ledger_id="L1"),
        ]
    )
    expected = ExpectedResult(behavior="answer", values=[1325.97, 1613.01], tolerance=0.01)
    result = numeric_correctness.score(trace, expected)
    assert result.passed is True


def test_numeric_correctness_not_applicable_for_non_answer_cases() -> None:
    """A clarify/refuse expectation has no numeric claim to check, so this must
    report 'not applicable' rather than pass or fail."""
    trace = _trace(route="clarify")
    expected = ExpectedResult(behavior="clarify")
    result = numeric_correctness.score(trace, expected)
    assert result.passed is None


# --- groundedness ---


def test_groundedness_passes_when_self_reported_matches_expected() -> None:
    """The self-reported groundedness_ok scorer must pass when it matches expected."""
    trace = _trace(groundedness_ok=True, claims=[])
    expected = ExpectedResult(behavior="answer", value=1.0, groundedness_ok=True)
    results = groundedness.score(trace, expected)
    assert results["groundedness_ok"].passed is True


def test_groundedness_self_reported_fails_when_it_disagrees_with_expected() -> None:
    """The self-reported groundedness_ok scorer must fail when it disagrees with expected."""
    trace = _trace(groundedness_ok=False, claims=[])
    expected = ExpectedResult(behavior="answer", value=1.0, groundedness_ok=True)
    results = groundedness.score(trace, expected)
    assert results["groundedness_ok"].passed is False


def test_harness_groundedness_independently_catches_an_ungrounded_claim() -> None:
    """Even if the agent self-reports groundedness_ok=True, the harness's own
    verify() call must independently catch a claim with no supporting ledger value."""
    trace = _trace(
        groundedness_ok=True,  # agent (wrongly) thinks it's fine
        ledger=[_ledger_entry("L1", "aggregate_spending_tool", {}, {"total": "100.00"})],
        claims=[Claim(value=Decimal("999.99"), ledger_id="L1")],
    )
    expected = ExpectedResult(behavior="answer", value=999.99, groundedness_ok=True)
    results = groundedness.score(trace, expected)
    assert results["groundedness_ok"].passed is True  # agrees with (wrong) self-report
    assert results["harness_groundedness_ok"].passed is False  # harness catches it anyway


def test_groundedness_not_applicable_for_non_answer_cases() -> None:
    """A refuse case has no claims/ledger to check groundedness against."""
    trace = _trace(route="refuse")
    expected = ExpectedResult(behavior="refuse")
    results = groundedness.score(trace, expected)
    assert results["groundedness_ok"].passed is None
    assert results["harness_groundedness_ok"].passed is None


# --- contextual_correctness ---


def test_contextual_correctness_passes_when_category_matches_in_ledger_args() -> None:
    """A matching category in the ledger's tool args must pass."""
    trace = _trace(
        ledger=[
            _ledger_entry(
                "L1",
                "aggregate_spending_tool",
                {"filt": {"category": "Groceries", "date_range": {"start": "2025-01-01"}}},
                {"total": "1613.01"},
            )
        ]
    )
    expected = ExpectedResult(
        behavior="answer", value=1613.01, expected_context={"category": "Groceries"}
    )
    result = contextual_correctness.score(trace, expected)
    assert result.passed is True


def test_contextual_correctness_fails_when_category_does_not_match() -> None:
    """A mismatched category (the follow-up resolved to the wrong scope) must fail."""
    trace = _trace(
        ledger=[
            _ledger_entry(
                "L1", "aggregate_spending_tool", {"filt": {"category": "Dining"}}, {"total": "1.0"}
            )
        ]
    )
    expected = ExpectedResult(
        behavior="answer", value=1.0, expected_context={"category": "Groceries"}
    )
    result = contextual_correctness.score(trace, expected)
    assert result.passed is False


def test_contextual_correctness_matches_nested_date_range() -> None:
    """A nested date_range dict in expected_context must match a nested date_range
    in the ledger args, even though the tool's arg structure isn't known to the scorer."""
    trace = _trace(
        ledger=[
            _ledger_entry(
                "L1",
                "aggregate_spending_tool",
                {"filt": {"date_range": {"start": "2025-01-01", "end": "2025-03-31"}}},
                {"total": "1.0"},
            )
        ]
    )
    expected = ExpectedResult(
        behavior="answer",
        value=1.0,
        expected_context={"date_range": {"start": "2025-01-01", "end": "2025-03-31"}},
    )
    result = contextual_correctness.score(trace, expected)
    assert result.passed is True


def test_contextual_correctness_not_applicable_with_no_expected_context() -> None:
    """A case that declares no expected_context has nothing for this scorer to check."""
    trace = _trace(ledger=[_ledger_entry("L1", "t", {}, {})])
    expected = ExpectedResult(behavior="answer", value=1.0)
    result = contextual_correctness.score(trace, expected)
    assert result.passed is None


def test_contextual_correctness_not_applicable_with_empty_ledger() -> None:
    """An empty ledger (no tool calls made) has no args to check expected_context against."""
    trace = _trace(ledger=[])
    expected = ExpectedResult(behavior="answer", value=1.0, expected_context={"category": "X"})
    result = contextual_correctness.score(trace, expected)
    assert result.passed is None


# --- clarification ---


def test_clarification_passes_when_expected_and_actual_both_clarify() -> None:
    """route == 'clarify' matching an expected 'clarify' must pass."""
    trace = _trace(route="clarify")
    expected = ExpectedResult(behavior="clarify")
    assert clarification.score(trace, expected).passed is True


def test_clarification_fails_on_unwanted_clarification() -> None:
    """An agent that clarifies when it shouldn't must fail this scorer, not be
    silently excluded."""
    trace = _trace(route="clarify")
    expected = ExpectedResult(behavior="answer", value=1.0)
    assert clarification.score(trace, expected).passed is False


def test_clarification_fails_when_agent_should_have_clarified_but_did_not() -> None:
    """An agent that guesses instead of clarifying must fail this scorer."""
    trace = _trace(route="answer")
    expected = ExpectedResult(behavior="clarify")
    assert clarification.score(trace, expected).passed is False


# --- refusal ---


def test_refusal_passes_when_expected_and_actual_both_refuse() -> None:
    """route == 'refuse' matching an expected 'refuse' must pass."""
    trace = _trace(route="refuse")
    expected = ExpectedResult(behavior="refuse")
    assert refusal.score(trace, expected).passed is True


def test_refusal_fails_on_unwanted_refusal() -> None:
    """An agent that refuses a question it should have answered must fail this scorer."""
    trace = _trace(route="refuse")
    expected = ExpectedResult(behavior="answer", value=1.0)
    assert refusal.score(trace, expected).passed is False


# --- efficiency ---


def test_efficiency_passes_when_actual_meets_optimal() -> None:
    """Using exactly the optimal number of tool calls must pass with zero overage."""
    trace = _trace(ledger=[_ledger_entry("L1", "t", {}, {})])
    expected = ExpectedResult(behavior="answer", value=1.0, optimal_tool_calls=1)
    result = efficiency.score(trace, expected)
    assert result.efficient.passed is True
    assert result.overage == 0
    assert result.actual_tool_calls == 1


def test_efficiency_fails_and_reports_overage_when_actual_exceeds_optimal() -> None:
    """Using more than the optimal number of tool calls must fail and report the overage."""
    trace = _trace(
        ledger=[
            _ledger_entry("L1", "list_categories_tool", {}, []),
            _ledger_entry("L2", "aggregate_spending_tool", {}, {}),
            _ledger_entry("L3", "aggregate_spending_tool", {}, {}),
        ]
    )
    expected = ExpectedResult(behavior="answer", value=1.0, optimal_tool_calls=1)
    result = efficiency.score(trace, expected)
    assert result.efficient.passed is False
    assert result.overage == 2
    assert result.actual_tool_calls == 3


def test_efficiency_not_applicable_with_no_optimal_declared() -> None:
    """A clarify/refuse case declares no optimal_tool_calls, so efficiency isn't scored."""
    trace = _trace(route="clarify", ledger=[])
    expected = ExpectedResult(behavior="clarify")
    result = efficiency.score(trace, expected)
    assert result.efficient.passed is None
    assert result.overage is None


def test_efficiency_reports_retries_separately_from_tool_call_overage() -> None:
    """Groundedness retries must be reported alongside efficiency but not affect
    the tool-call efficiency verdict itself."""
    trace = _trace(ledger=[_ledger_entry("L1", "t", {}, {})], retries=2)
    expected = ExpectedResult(behavior="answer", value=1.0, optimal_tool_calls=1)
    result = efficiency.score(trace, expected)
    assert result.retries == 2
    assert result.efficient.passed is True
