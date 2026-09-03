"""Tests for detect_trend and detect_subscription_changes."""

from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import Engine

from finance_qna.tools.models import DateRange
from finance_qna.tools.trend_tools import detect_subscription_changes, detect_trend


def test_detect_subscription_changes_finds_labeled_event(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """The labeled StreamFlix price change must be detected with the right amounts and date."""
    event = ground_truth["subscription_change_event"]
    reference_date = date.fromisoformat(ground_truth["reference_date"])
    full_range = DateRange(start=date(2000, 1, 1), end=reference_date)

    changes = detect_subscription_changes(fixture_engine, full_range)
    matching = [c for c in changes if c.merchant == event["merchant"]]

    assert len(matching) == 1
    detected = matching[0]
    assert detected.old_amount == Decimal(event["old_amount"])
    assert detected.new_amount == Decimal(event["new_amount"])
    assert detected.change_date.isoformat() >= event["change_month"]


def test_detect_subscription_changes_ignores_stable_subscriptions(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """A subscription whose amount never changes must not appear in the results."""
    reference_date = date.fromisoformat(ground_truth["reference_date"])
    full_range = DateRange(start=date(2000, 1, 1), end=reference_date)

    changes = detect_subscription_changes(fixture_engine, full_range)
    merchants_with_changes = {c.merchant for c in changes}

    assert "GymPass" not in merchants_with_changes
    assert "CloudDrive+" not in merchants_with_changes


def test_detect_trend_shows_increasing_direction_for_subscriptions(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """Subscriptions spend must trend upward across the labeled price-change month."""
    reference_date = date.fromisoformat(ground_truth["reference_date"])
    full_range = DateRange(start=date(2000, 1, 1), end=reference_date)

    trend = detect_trend(fixture_engine, "Subscriptions", full_range, granularity="month")

    assert trend.direction == "increasing"
    assert trend.pct_change is not None
    assert trend.pct_change > 0


def test_detect_trend_series_length_matches_num_months(
    fixture_engine: Engine, ground_truth: dict[str, Any]
) -> None:
    """A full-range monthly trend must have one series point per generated month."""
    reference_date = date.fromisoformat(ground_truth["reference_date"])
    full_range = DateRange(start=date(2000, 1, 1), end=reference_date)

    trend = detect_trend(fixture_engine, "Subscriptions", full_range, granularity="month")

    assert len(trend.series) == ground_truth["num_months"]
