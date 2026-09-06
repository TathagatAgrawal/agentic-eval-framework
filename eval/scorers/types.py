"""The shared result type every scorer returns."""

from pydantic import BaseModel


class ScoreResult(BaseModel):
    """The outcome of checking one metric against one turn's `RunTrace`.

    `passed=None` means "not applicable" -- e.g. a clarify/refuse turn has no
    numeric claims to check, or a test case declares no `expected_context` to
    verify. Not-applicable results are excluded from pass-rate aggregation
    rather than counted as either a pass or a fail.
    """

    metric: str
    passed: bool | None
    detail: str = ""
