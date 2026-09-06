"""Scores groundedness two ways: the agent's self-reported flag, and an
independent, harness-computed check -- see design/eval-harness-plan.md §2.

`harness_groundedness_ok` calls `finance_qna.agent.groundedness.verify()`
directly, the same pure function the agent itself uses, so it stays valid as
an apples-to-apples measure once a second architecture (with its own,
possibly different, self-reporting) exists.
"""

from eval.schema import ExpectedResult
from eval.scorers.types import ScoreResult
from finance_qna.agent.groundedness import verify
from finance_qna.tracing.trace import RunTrace


def score(trace: RunTrace, expected: ExpectedResult) -> dict[str, ScoreResult]:
    """Return both 'groundedness_ok' (self-reported) and 'harness_groundedness_ok'
    (independently recomputed), each checked against `expected.groundedness_ok`."""
    if expected.behavior != "answer":
        not_applicable = "not an answer case"
        return {
            "groundedness_ok": ScoreResult(
                metric="groundedness_ok", passed=None, detail=not_applicable
            ),
            "harness_groundedness_ok": ScoreResult(
                metric="harness_groundedness_ok", passed=None, detail=not_applicable
            ),
        }

    self_reported_passed = trace.groundedness_ok == expected.groundedness_ok

    harness_ok = (
        verify(trace.draft_answer, trace.ledger).ok if trace.draft_answer is not None else False
    )
    harness_passed = harness_ok == expected.groundedness_ok

    return {
        "groundedness_ok": ScoreResult(
            metric="groundedness_ok",
            passed=self_reported_passed,
            detail=""
            if self_reported_passed
            else f"expected {expected.groundedness_ok}, got {trace.groundedness_ok}",
        ),
        "harness_groundedness_ok": ScoreResult(
            metric="harness_groundedness_ok",
            passed=harness_passed,
            detail=""
            if harness_passed
            else f"expected {expected.groundedness_ok}, got {harness_ok}",
        ),
    }
