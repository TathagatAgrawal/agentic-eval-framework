"""Scores whether a multi-turn follow-up actually resolved to the right
category/date_range/etc., by checking the turn's ledger args -- not just that
some answer came back.

Uses a generic flatten-and-partial-match approach (see
design/eval-harness-plan.md §4.3) rather than being tool-aware, so it works
across every tool's differently-shaped args without the scorer needing to know
each tool's schema.
"""

from typing import Any

from eval.schema import ExpectedResult
from eval.scorers.types import ScoreResult
from finance_qna.tracing.trace import RunTrace


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted-path keys, e.g.
    `{"filt": {"category": "X"}}` -> `{"filt.category": "X"}`."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, sub_value in value.items():
        sub_prefix = f"{prefix}.{key}" if prefix else key
        flat.update(_flatten(sub_value, sub_prefix))
    return flat


def score(trace: RunTrace, expected: ExpectedResult) -> ScoreResult:
    """Pass iff every key/value in `expected.expected_context` is found (matched
    by dotted-path suffix, so nesting depth doesn't have to match exactly) in the
    flattened args of some ledger entry from this turn."""
    if not expected.expected_context:
        return ScoreResult(
            metric="contextual_correctness", passed=None, detail="no expected_context declared"
        )
    if not trace.ledger:
        return ScoreResult(
            metric="contextual_correctness", passed=None, detail="no ledger entries to check"
        )

    expected_flat = _flatten(expected.expected_context)
    ledger_flats = [_flatten(entry["args"]) for entry in trace.ledger]

    unmatched = [
        f"{key}={value!r}"
        for key, value in expected_flat.items()
        if not any(
            str(actual_value) == str(value)
            for flat in ledger_flats
            for actual_key, actual_value in flat.items()
            if actual_key.endswith(key)
        )
    ]

    passed = not unmatched
    detail = "" if passed else f"not found in any ledger entry's args: {unmatched}"
    return ScoreResult(metric="contextual_correctness", passed=passed, detail=detail)
