"""Time-series trend detection and recurring-subscription price-change detection.

Both tools compute their results server-side (SQL grouping + Python aggregation)
rather than handing raw transactions to the LLM to reason about, per the design
decision to keep arithmetic the eval harness must check exactly out of the LLM's
hands.
"""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Literal

from sqlalchemy import Engine

from finance_qna.data.schema import transactions
from finance_qna.tools.models import (
    DateRange,
    SubscriptionChangeEvent,
    TransactionFilter,
    TrendPoint,
    TrendResult,
)
from finance_qna.tools.query_tools import filter_conditions, joined_transactions, validate_category

FLAT_THRESHOLD_PCT = Decimal("2")  # below this |% change| between first and last period, "flat"


def _quarter_label(month: int, year: int) -> str:
    """Return a "YYYY-Q#" label for the given calendar month/year."""
    quarter = (month - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def detect_trend(
    engine: Engine,
    category: str | None,
    date_range: DateRange,
    granularity: Literal["month", "quarter"] = "month",
) -> TrendResult:
    """Compute a spending time series for `category` (or all categories) over
    `date_range`, bucketed by month or quarter, plus its overall direction."""
    if category is not None:
        validate_category(engine, category)

    filt = TransactionFilter(category=category, date_range=date_range)
    query = joined_transactions().where(*filter_conditions(filt))
    with engine.connect() as conn:
        rows = conn.execute(query).all()

    buckets: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for row in rows:
        key = (
            row.date.strftime("%Y-%m")
            if granularity == "month"
            else _quarter_label(row.date.month, row.date.year)
        )
        buckets[key] += row.amount

    series = [TrendPoint(period=k, total=v) for k, v in sorted(buckets.items())]

    if len(series) < 2 or series[0].total == 0:
        direction: Literal["increasing", "decreasing", "flat"] = "flat"
        pct_change: Decimal | None = None
    else:
        first, last = series[0].total, series[-1].total
        pct_change = (last - first) / first * Decimal("100")
        if pct_change > FLAT_THRESHOLD_PCT:
            direction = "increasing"
        elif pct_change < -FLAT_THRESHOLD_PCT:
            direction = "decreasing"
        else:
            direction = "flat"

    return TrendResult(
        category=category,
        granularity=granularity,
        series=series,
        direction=direction,
        pct_change=pct_change,
    )


def detect_subscription_changes(
    engine: Engine, date_range: DateRange
) -> list[SubscriptionChangeEvent]:
    """Find every recurring subscription merchant whose amount changed within `date_range`.

    A "change" is any two consecutive subscription transactions from the same
    merchant with different amounts.
    """
    filt = TransactionFilter(category=None, date_range=date_range, account=None, merchant=None)
    query = joined_transactions().where(
        *filter_conditions(filt), transactions.c.is_subscription.is_(True)
    )
    with engine.connect() as conn:
        rows = conn.execute(query).all()

    by_merchant: dict[str, list[tuple[date, Decimal]]] = defaultdict(list)
    for row in rows:
        by_merchant[row.merchant].append((row.date, row.amount))

    events: list[SubscriptionChangeEvent] = []
    for merchant, txns in by_merchant.items():
        txns.sort(key=lambda t: t[0])
        for (_prev_date, prev_amount), (curr_date, curr_amount) in zip(
            txns, txns[1:], strict=False
        ):
            if curr_amount != prev_amount:
                events.append(
                    SubscriptionChangeEvent(
                        merchant=merchant,
                        old_amount=prev_amount,
                        new_amount=curr_amount,
                        change_date=curr_date,
                    )
                )

    return events
