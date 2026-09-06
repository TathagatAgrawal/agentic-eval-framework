"""Scores whether the agent's stated number(s) match the expected value(s).

Reads `trace.draft_answer.claims` directly (already parsed as `Decimal` by the
agent) rather than regex-parsing the final answer text -- see
design/eval-harness-plan.md's opening note on why this is possible at all.
"""

from decimal import Decimal

from eval.schema import ExpectedResult
from eval.scorers.types import ScoreResult
from finance_qna.tracing.trace import RunTrace


def score(trace: RunTrace, expected: ExpectedResult) -> ScoreResult:
    """Pass iff every expected value is matched, within tolerance, by some claim."""
    if expected.behavior != "answer":
        return ScoreResult(metric="numeric_correctness", passed=None, detail="not an answer case")

    targets = expected.values if expected.values is not None else [expected.value]
    target_decimals = [Decimal(str(v)) for v in targets if v is not None]
    tolerance = Decimal(str(expected.tolerance))

    claim_values = (
        [claim.value for claim in trace.draft_answer.claims] if trace.draft_answer else []
    )

    unmatched = [
        target
        for target in target_decimals
        if not any(abs(claim_value - target) <= tolerance for claim_value in claim_values)
    ]

    passed = not unmatched
    detail = "" if passed else f"no claim matched expected value(s): {unmatched}"
    return ScoreResult(metric="numeric_correctness", passed=passed, detail=detail)
