"""Scores whether the agent asked for clarification exactly when it should have
-- and, just as important, didn't when it shouldn't have."""

from eval.schema import ExpectedResult
from eval.scorers.types import ScoreResult
from finance_qna.tracing.trace import RunTrace


def score(trace: RunTrace, expected: ExpectedResult) -> ScoreResult:
    """Pass iff (expected.behavior == 'clarify') matches (trace.route == 'clarify')."""
    expected_clarify = expected.behavior == "clarify"
    actual_clarify = trace.route == "clarify"
    passed = expected_clarify == actual_clarify
    detail = (
        ""
        if passed
        else f"expected route=='clarify' to be {expected_clarify}, got route={trace.route!r}"
    )
    return ScoreResult(metric="clarification", passed=passed, detail=detail)
