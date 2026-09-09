"""Drives an `AgentAdapter` against the test set and scores the results.

Depends only on `AgentAdapter`/`RunTrace` (see design/eval-harness-plan.md §2)
and the scorers -- never on LangGraph, Gemini, or any agent-internal type.
Every scorer runs unconditionally for every turn; each one already reports
`passed=None` ("not applicable") for cases it doesn't apply to, so the runner
doesn't need its own per-behavior special-casing.
"""

import time
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from eval.results import RunRecord, TestCaseResult
from eval.schema import ExpectedResult, TestCase, Turn
from eval.scorers import (
    clarification,
    contextual_correctness,
    efficiency,
    groundedness,
    numeric_correctness,
    refusal,
)
from eval.scorers.efficiency import EfficiencyResult
from eval.scorers.types import ScoreResult
from eval.store import save_run
from finance_qna.agent.adapter import AgentAdapter
from finance_qna.tracing.trace import RunTrace, write_trace


def score_turn(
    trace: RunTrace, expected: ExpectedResult
) -> tuple[dict[str, ScoreResult], EfficiencyResult]:
    """Run every scorer against one turn's trace, returning per-metric results
    plus the efficiency detail (which carries numeric fields beyond a bool)."""
    metric_results: dict[str, ScoreResult] = {
        "numeric_correctness": numeric_correctness.score(trace, expected),
        "contextual_correctness": contextual_correctness.score(trace, expected),
        "clarification": clarification.score(trace, expected),
        "refusal": refusal.score(trace, expected),
    }
    metric_results.update(groundedness.score(trace, expected))

    efficiency_result = efficiency.score(trace, expected)
    metric_results["efficient"] = efficiency_result.efficient

    return metric_results, efficiency_result


def _turns_for(case: TestCase) -> list[Turn]:
    """Return a multi-turn case's turns, or a single synthesized Turn wrapping a
    single-turn case's question/expected -- so callers can treat both uniformly."""
    if case.turns is not None:
        return case.turns
    assert case.question is not None
    assert case.expected is not None
    return [Turn(question=case.question, expected=case.expected)]


def _sum_or_none(values: Iterable[int | None]) -> int | None:
    """Sum a sequence of optional ints, returning None if any value is None."""
    materialized = list(values)
    if any(v is None for v in materialized):
        return None
    return sum(v for v in materialized if v is not None)


def _aggregate_scores(per_turn_metrics: list[dict[str, ScoreResult]]) -> dict[str, bool | None]:
    """Combine each turn's per-metric results into one pass/fail per metric for
    the whole case: passes only if every turn where that metric applied passed."""
    all_metric_names = {name for turn_metrics in per_turn_metrics for name in turn_metrics}
    aggregated: dict[str, bool | None] = {}
    for name in all_metric_names:
        applicable = [
            turn_metrics[name].passed
            for turn_metrics in per_turn_metrics
            if name in turn_metrics and turn_metrics[name].passed is not None
        ]
        aggregated[name] = all(applicable) if applicable else None
    return aggregated


def run_test_case(adapter: AgentAdapter, case: TestCase, run_dir: Path) -> TestCaseResult:
    """Run one test case (single-turn or multi-turn) against `adapter` and score it.

    If a turn raises (e.g. a 429 from the model provider), the case stops there
    -- later turns can't run anyway, since they'd need this one's trace as
    `prior_turns` -- but every turn that *did* complete is still scored and
    returned, with `error` set to what stopped it, rather than losing that
    turn's work.
    """
    case_trace_dir = run_dir / case.id
    prior_turns: list[RunTrace] = []
    per_turn_metrics: list[dict[str, ScoreResult]] = []
    per_turn_efficiency: list[EfficiencyResult] = []
    trace_refs: list[str] = []
    error: str | None = None

    for turn in _turns_for(case):
        try:
            trace = adapter.run_turn(turn.question, prior_turns=prior_turns)
        except Exception as exc:  # deliberately broad: isolate this case's failure from the rest
            error = f"{type(exc).__name__}: {exc}"
            break

        trace_path = write_trace(trace, case_trace_dir)
        trace_refs.append(str(trace_path))

        metric_results, turn_efficiency = score_turn(trace, turn.expected)
        per_turn_metrics.append(metric_results)
        per_turn_efficiency.append(turn_efficiency)

        prior_turns.append(trace)

    return TestCaseResult(
        case_id=case.id,
        category=case.category,
        scores=_aggregate_scores(per_turn_metrics),
        optimal_tool_calls=(
            _sum_or_none(e.optimal_tool_calls for e in per_turn_efficiency)
            if per_turn_efficiency
            else None
        ),
        actual_tool_calls=sum(e.actual_tool_calls for e in per_turn_efficiency),
        overage=(
            _sum_or_none(e.overage for e in per_turn_efficiency) if per_turn_efficiency else None
        ),
        retries=sum(e.retries for e in per_turn_efficiency),
        trace_refs=trace_refs,
        error=error,
    )


def _pass_rates(results: list[TestCaseResult]) -> dict[str, float]:
    """Compute the pass rate per metric across `results`, excluding
    not-applicable (None) scores from the denominator."""
    metric_names = {name for result in results for name in result.scores}
    rates: dict[str, float] = {}
    for name in metric_names:
        applicable = [
            result.scores[name] for result in results if result.scores.get(name) is not None
        ]
        if applicable:
            rates[name] = sum(1 for v in applicable if v) / len(applicable)
    return rates


def _summarize(results: list[TestCaseResult]) -> dict[str, float]:
    """Build the overall and per-category pass-rate summary for a RunRecord."""
    summary = {f"overall.{name}": rate for name, rate in _pass_rates(results).items()}

    by_category: dict[str, list[TestCaseResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    for category, category_results in by_category.items():
        for name, rate in _pass_rates(category_results).items():
            summary[f"{category}.{name}"] = rate

    return summary


def _errored_result(case: TestCase, exc: Exception) -> TestCaseResult:
    """Build a placeholder TestCaseResult for a case that failed before
    run_test_case could return one at all (belt-and-braces on top of
    run_test_case's own per-turn error handling)."""
    return TestCaseResult(
        case_id=case.id,
        category=case.category,
        scores={},
        optimal_tool_calls=None,
        actual_tool_calls=0,
        overage=None,
        retries=0,
        trace_refs=[],
        error=f"{type(exc).__name__}: {exc}",
    )


def run_eval(
    adapter: AgentAdapter,
    cases: list[TestCase],
    run_dir: Path,
    label: str = "",
    run_id: str | None = None,
    runs_dir: Path | None = None,
    delay_seconds: float = 0.0,
) -> RunRecord:
    """Run every test case against `adapter`, scoring each, and aggregate a `RunRecord`.

    `run_id` defaults to a fresh random id; callers that need to know the id
    before traces are written (e.g. to name `run_dir` after it) can generate
    one themselves and pass it in.

    If `runs_dir` is given, the `RunRecord` is saved after *every* case, not
    just once at the end -- a case that raises (a 429, or anything else) never
    loses the cases that already completed, and the saved summary always
    reflects whatever's actually on disk. `run_test_case` already isolates a
    failure to a single turn/case internally; this is the second, outer layer
    of that same guarantee, for anything that escapes it.

    `delay_seconds`, if positive, is slept between cases (never before the
    first) to stay under a model provider's per-minute rate limit during a
    live run. 0 (the default) disables it.
    """
    resolved_run_id = run_id or uuid.uuid4().hex[:8]
    results: list[TestCaseResult] = []
    record = RunRecord(
        run_id=resolved_run_id,
        timestamp=datetime.now(UTC),
        agent_id=adapter.id,
        label=label,
        results=results,
        summary={},
    )

    for index, case in enumerate(cases):
        if delay_seconds > 0 and index > 0:
            time.sleep(delay_seconds)
        try:
            result = run_test_case(adapter, case, run_dir)
        except Exception as exc:
            result = _errored_result(case, exc)
        results.append(result)

        record = record.model_copy(
            update={
                "timestamp": datetime.now(UTC),
                "results": list(results),
                "summary": _summarize(results),
            }
        )
        if runs_dir is not None:
            save_run(record, runs_dir=runs_dir)

    return record
