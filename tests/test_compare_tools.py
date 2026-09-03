"""Tests for compare_periods."""

from calendar import monthrange
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import Engine

from finance_qna.tools.compare_tools import compare_periods
from finance_qna.tools.models import DateRange


def _month_range(iso_month: str) -> DateRange:
    """Build a DateRange spanning the full calendar month for a "YYYY-MM-01" string."""
    month_start = date.fromisoformat(iso_month)
    last_day = monthrange(month_start.year, month_start.month)[1]
    return DateRange(start=month_start, end=date(month_start.year, month_start.month, last_day))


def test_compare_periods_matches_ground_truth_diff(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """diff and totals must equal the difference between two ground-truth monthly totals."""
    category = "Dining"
    months = sorted(ground_truth["monthly_category_totals"][category])
    month_a, month_b = months[0], months[1]

    result = compare_periods(fixture_engine, category, _month_range(month_a), _month_range(month_b))

    expected_a = Decimal(ground_truth["monthly_category_totals"][category][month_a])
    expected_b = Decimal(ground_truth["monthly_category_totals"][category][month_b])

    assert result.total_a == expected_a
    assert result.total_b == expected_b
    assert result.diff == expected_b - expected_a


def test_compare_periods_pct_change_direction(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """Comparing the Subscriptions category across the labeled price-change month
    must show a positive percentage increase."""
    event = ground_truth["subscription_change_event"]
    before_month = date.fromisoformat(event["change_month"])
    before_month = date(before_month.year, before_month.month - 1, 1)
    after_month = date.fromisoformat(event["change_month"])

    result = compare_periods(
        fixture_engine,
        "Subscriptions",
        _month_range(before_month.isoformat()),
        _month_range(after_month.isoformat()),
    )

    assert result.pct_change is not None
    assert result.pct_change > 0
