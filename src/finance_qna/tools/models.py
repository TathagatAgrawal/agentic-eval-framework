"""Pydantic argument and result schemas shared by every tool.

Every LLM-facing tool argument and result is a model defined here, so tool calls
are normalized, comparable objects (tool name + validated args) rather than free
text -- this is what lets the eval harness diff a tool call against a labeled
expected call, and what lets the groundedness checker walk results by known field
names instead of doing generic dict traversal.
"""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, model_validator


class DateRange(BaseModel):
    """An inclusive date range, e.g. the first through last day of a month."""

    start: date
    end: date

    @model_validator(mode="after")
    def _check_order(self) -> "DateRange":
        """Reject a range where the end date precedes the start date."""
        if self.end < self.start:
            raise ValueError(f"date_range end ({self.end}) is before start ({self.start})")
        return self


class TransactionFilter(BaseModel):
    """Filter criteria for selecting a set of transactions.

    `category`, `account`, and `merchant` are matched by exact name against the
    dataset's live vocabulary (validated by the tool functions, not this model,
    since that check requires a database).
    """

    category: str | None = None
    date_range: DateRange
    account: str | None = None
    merchant: str | None = None
    min_amount: Decimal | None = None
    max_amount: Decimal | None = None


class TransactionRow(BaseModel):
    """A single transaction as returned by `get_transactions`."""

    transaction_id: int
    date: date
    amount: Decimal
    description: str
    category: str
    merchant: str
    account: str


class AggregateResult(BaseModel):
    """The result of `aggregate_spending` for one group (or the whole filter, when
    `group_by="none"`)."""

    total: Decimal
    count: int
    group_key: str | None = None
    filter_applied: TransactionFilter


class CompareResult(BaseModel):
    """The result of comparing total spend between two periods, optionally scoped
    to one category."""

    category: str | None
    period_a: DateRange
    period_b: DateRange
    total_a: Decimal
    total_b: Decimal
    diff: Decimal
    pct_change: Decimal | None


class TrendPoint(BaseModel):
    """One point in a spending time series: a period label and its total."""

    period: str
    total: Decimal


class TrendResult(BaseModel):
    """The result of `detect_trend`: a time series plus its overall direction."""

    category: str | None
    granularity: Literal["month", "quarter"]
    series: list[TrendPoint]
    direction: Literal["increasing", "decreasing", "flat"]
    pct_change: Decimal | None


class SubscriptionChangeEvent(BaseModel):
    """A detected change in a recurring subscription's amount."""

    merchant: str
    old_amount: Decimal
    new_amount: Decimal
    change_date: date
