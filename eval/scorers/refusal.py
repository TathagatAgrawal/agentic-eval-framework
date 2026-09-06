"""Scores whether the agent declined an out-of-scope question -- and, just as
important, didn't refuse a question it should have answered."""

from eval.schema import ExpectedResult
from eval.scorers.types import ScoreResult
from finance_qna.tracing.trace import RunTrace


def score(trace: RunTrace, expected: ExpectedResult) -> ScoreResult:
    """Pass iff (expected.behavior == 'refuse') matches (trace.route == 'refuse')."""
    expected_refuse = expected.behavior == "refuse"
    actual_refuse = trace.route == "refuse"
    passed = expected_refuse == actual_refuse
    detail = (
        ""
        if passed
        else f"expected route=='refuse' to be {expected_refuse}, got route={trace.route!r}"
    )
    return ScoreResult(metric="refusal", passed=passed, detail=detail)
