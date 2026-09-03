"""Tests for list_categories, get_transactions, and aggregate_spending."""

from calendar import monthrange
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import Engine

from finance_qna.tools.errors import CategoryNotFoundError
from finance_qna.tools.models import AggregateResult, DateRange, TransactionFilter
from finance_qna.tools.query_tools import aggregate_spending, get_transactions, list_categories


def test_list_categories_includes_expected_names(fixture_engine: Engine) -> None:
    """The fixed category vocabulary used by the generator must all be present."""
    names = list_categories(fixture_engine)
    assert names == sorted(names)
    for expected in ["Dining", "Groceries", "Subscriptions", "Shopping", "Income"]:
        assert expected in names


def test_aggregate_spending_matches_ground_truth_for_one_month(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """A single-month, single-category aggregate must equal the ground-truth total."""
    category = "Dining"
    month = next(iter(ground_truth["monthly_category_totals"][category]))
    month_start = date.fromisoformat(month)
    last_day = monthrange(month_start.year, month_start.month)[1]
    month_end = date(month_start.year, month_start.month, last_day)

    filt = TransactionFilter(
        category=category, date_range=DateRange(start=month_start, end=month_end)
    )
    result = aggregate_spending(fixture_engine, filt)

    assert isinstance(result, AggregateResult)
    expected_total = Decimal(ground_truth["monthly_category_totals"][category][month])
    assert result.total == expected_total


def test_aggregate_spending_group_by_month_matches_ground_truth(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """Grouping by month over the full range must reproduce every monthly total."""
    category = "Groceries"
    reference_date = date.fromisoformat(ground_truth["reference_date"])
    full_range = DateRange(start=date(2000, 1, 1), end=reference_date)
    filt = TransactionFilter(category=category, date_range=full_range)

    results = aggregate_spending(fixture_engine, filt, group_by="month")
    assert isinstance(results, list)

    actual_by_month = {r.group_key: r.total for r in results}
    expected_by_month = {
        month[:7]: Decimal(total)  # ground truth keys are "YYYY-MM-01"; normalize to "YYYY-MM"
        for month, total in ground_truth["monthly_category_totals"][category].items()
    }
    assert actual_by_month == expected_by_month


def test_aggregate_spending_rejects_unknown_category(fixture_engine: Engine) -> None:
    """An unknown category must raise a structured error listing valid categories."""
    filt = TransactionFilter(
        category="Not A Real Category",
        date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31)),
    )
    with pytest.raises(CategoryNotFoundError) as exc_info:
        aggregate_spending(fixture_engine, filt)

    assert exc_info.value.category == "Not A Real Category"
    assert "Dining" in exc_info.value.valid_categories


def test_get_transactions_respects_limit(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """get_transactions must never return more rows than the requested limit."""
    reference_date = date.fromisoformat(ground_truth["reference_date"])
    filt = TransactionFilter(date_range=DateRange(start=date(2000, 1, 1), end=reference_date))
    rows = get_transactions(fixture_engine, filt, limit=5)
    assert len(rows) == 5
