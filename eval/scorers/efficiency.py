"""Scores tool-call efficiency against the test case's labeled optimal count
(see design/eval-harness-plan.md §4.1/§6), not just a raw, unscored count."""

from pydantic import BaseModel

from eval.schema import ExpectedResult
from eval.scorers.types import ScoreResult
from finance_qna.tracing.trace import RunTrace


class EfficiencyResult(BaseModel):
    """Efficiency detail for one turn: actual vs. optimal tool calls, plus retries."""

    optimal_tool_calls: int | None
    actual_tool_calls: int
    overage: int | None
    retries: int
    efficient: ScoreResult


def score(trace: RunTrace, expected: ExpectedResult) -> EfficiencyResult:
    """Compare `len(trace.ledger)` to `expected.optimal_tool_calls`.

    Not scored (efficient.passed=None) if the test case declares no
    optimal_tool_calls -- e.g. clarify/refuse turns, which make no tool calls
    by design and aren't measured on this axis.
    """
    actual = len(trace.ledger)

    if expected.optimal_tool_calls is None:
        return EfficiencyResult(
            optimal_tool_calls=None,
            actual_tool_calls=actual,
            overage=None,
            retries=trace.retries,
            efficient=ScoreResult(
                metric="efficient", passed=None, detail="no optimal_tool_calls declared"
            ),
        )

    overage = actual - expected.optimal_tool_calls
    passed = overage <= 0
    detail = (
        "" if passed else f"used {actual} tool calls, optimal was {expected.optimal_tool_calls}"
    )
    return EfficiencyResult(
        optimal_tool_calls=expected.optimal_tool_calls,
        actual_tool_calls=actual,
        overage=overage,
        retries=trace.retries,
        efficient=ScoreResult(metric="efficient", passed=passed, detail=detail),
    )
