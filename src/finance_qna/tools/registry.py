"""Binds the tool-layer functions to a specific database engine as LangChain tools.

The core functions in `query_tools`, `compare_tools`, and `trend_tools` all take
an `Engine` as their first argument so they stay unit-testable without LangChain.
This module wraps each one in a closure that hides the engine, so the LLM only
ever sees the arguments it's actually supposed to supply.
"""

from typing import Literal

from langchain_core.tools import BaseTool, tool
from sqlalchemy import Engine

from finance_qna.tools.compare_tools import compare_periods
from finance_qna.tools.models import (
    AggregateResult,
    CompareResult,
    DateRange,
    SubscriptionChangeEvent,
    TransactionFilter,
    TransactionRow,
    TrendResult,
)
from finance_qna.tools.query_tools import aggregate_spending, get_transactions, list_categories
from finance_qna.tools.trend_tools import detect_subscription_changes, detect_trend


def build_tools(engine: Engine) -> list[BaseTool]:
    """Return every financial-data tool, bound to `engine`, ready for `.bind_tools()`."""

    @tool
    def list_categories_tool() -> list[str]:
        """List every spending category name that exists in the dataset."""
        return list_categories(engine)

    @tool
    def get_transactions_tool(filt: TransactionFilter, limit: int = 50) -> list[TransactionRow]:
        """Return up to `limit` individual transactions matching the filter, most recent first.

        Use this for "show me" style questions or to inspect a small set of
        transactions, not for totals -- use aggregate_spending_tool for totals.
        """
        return get_transactions(engine, filt, limit)

    @tool
    def aggregate_spending_tool(
        filt: TransactionFilter,
        group_by: Literal["none", "category", "month"] = "none",
    ) -> AggregateResult | list[AggregateResult]:
        """Sum and count transactions matching a filter.

        This is the primary tool for "how much did I spend on X" questions. Set
        `group_by` to "category" or "month" to get a breakdown instead of one total.
        """
        return aggregate_spending(engine, filt, group_by)

    @tool
    def compare_periods_tool(
        category: str | None, period_a: DateRange, period_b: DateRange
    ) -> CompareResult:
        """Compare total spend between two date ranges, optionally scoped to one category.

        Use this for "did I spend more on X or Y" or "vs last year" style questions.
        """
        return compare_periods(engine, category, period_a, period_b)

    @tool
    def detect_trend_tool(
        category: str | None,
        date_range: DateRange,
        granularity: Literal["month", "quarter"] = "month",
    ) -> TrendResult:
        """Compute a spending time series and its overall direction (increasing,
        decreasing, or flat) for a category over a date range."""
        return detect_trend(engine, category, date_range, granularity)

    @tool
    def detect_subscription_changes_tool(date_range: DateRange) -> list[SubscriptionChangeEvent]:
        """Find every recurring subscription whose amount changed within a date range."""
        return detect_subscription_changes(engine, date_range)

    return [
        list_categories_tool,
        get_transactions_tool,
        aggregate_spending_tool,
        compare_periods_tool,
        detect_trend_tool,
        detect_subscription_changes_tool,
    ]
