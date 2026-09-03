"""Comparing total spend between two time periods."""

from decimal import Decimal

from sqlalchemy import Engine

from finance_qna.tools.models import AggregateResult, CompareResult, DateRange, TransactionFilter
from finance_qna.tools.query_tools import aggregate_spending


def compare_periods(
    engine: Engine,
    category: str | None,
    period_a: DateRange,
    period_b: DateRange,
) -> CompareResult:
    """Compare total spend in `period_a` vs `period_b`, optionally scoped to one category.

    `diff` and `pct_change` describe the change from period_a to period_b
    (positive `diff` means period_b was higher).
    """
    result_a = aggregate_spending(engine, TransactionFilter(category=category, date_range=period_a))
    result_b = aggregate_spending(engine, TransactionFilter(category=category, date_range=period_b))
    assert isinstance(result_a, AggregateResult)
    assert isinstance(result_b, AggregateResult)

    total_a, total_b = result_a.total, result_b.total
    diff = total_b - total_a
    pct_change = (diff / total_a * Decimal("100")) if total_a != 0 else None

    return CompareResult(
        category=category,
        period_a=period_a,
        period_b=period_b,
        total_a=total_a,
        total_b=total_b,
        diff=diff,
        pct_change=pct_change,
    )
