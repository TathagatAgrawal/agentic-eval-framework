"""Output schemas for one test case's scored result and one full eval run.

Kept separate from `eval/schema.py` (the test-case *input* schema) and from
`eval/store.py` (persistence, not yet built) so the runner can be built and
tested before the regression store exists, per the build order in
design/eval-harness-plan.md §10.
"""

from datetime import datetime

from pydantic import BaseModel


class TestCaseResult(BaseModel):
    """The scored outcome of one test case (single-turn, or an aggregated
    multi-turn sequence)."""

    case_id: str
    category: str
    scores: dict[str, bool | None]  # metric name -> pass/fail/not-applicable
    optimal_tool_calls: int | None
    actual_tool_calls: int
    overage: int | None
    retries: int
    trace_refs: list[str]


class RunRecord(BaseModel):
    """The full result of one eval run across the whole test set."""

    run_id: str
    timestamp: datetime
    agent_id: str
    label: str
    results: list[TestCaseResult]
    summary: dict[str, float]  # e.g. "overall.numeric_correctness", "trend.efficient" -> pass rate
