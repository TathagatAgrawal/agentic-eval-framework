"""Category listing, raw transaction lookup, and spending aggregation.

Every function here builds a parameterized SQLAlchemy query from validated
Pydantic args -- never string concatenation -- so SQL injection isn't possible
regardless of what ends up in free-text fields like `description`.
"""

from collections import defaultdict
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import Engine, Select, select
from sqlalchemy.sql import ColumnElement

from finance_qna.data.schema import accounts, categories, merchants, transactions
from finance_qna.tools.errors import CategoryNotFoundError
from finance_qna.tools.models import AggregateResult, TransactionFilter, TransactionRow


def list_categories(engine: Engine) -> list[str]:
    """Return every category name in the dataset, sorted alphabetically."""
    with engine.connect() as conn:
        rows = conn.execute(select(categories.c.name).order_by(categories.c.name)).all()
    return [row.name for row in rows]


def validate_category(engine: Engine, category: str) -> None:
    """Raise `CategoryNotFoundError` if `category` isn't a real category name."""
    valid = list_categories(engine)
    if category not in valid:
        raise CategoryNotFoundError(category, valid)


def joined_transactions() -> Select[Any]:
    """Build the base transactions query joined to account/merchant/category names."""
    return select(
        transactions.c.transaction_id,
        transactions.c.date,
        transactions.c.amount,
        transactions.c.description,
        categories.c.name.label("category"),
        merchants.c.name.label("merchant"),
        accounts.c.name.label("account"),
    ).select_from(
        transactions.join(categories, transactions.c.category_id == categories.c.category_id)
        .join(merchants, transactions.c.merchant_id == merchants.c.merchant_id)
        .join(accounts, transactions.c.account_id == accounts.c.account_id)
    )


def filter_conditions(filt: TransactionFilter) -> list[ColumnElement[bool]]:
    """Translate a `TransactionFilter` into a list of SQLAlchemy WHERE conditions."""
    conditions: list[ColumnElement[bool]] = [
        transactions.c.date >= filt.date_range.start,
        transactions.c.date <= filt.date_range.end,
    ]
    if filt.category is not None:
        conditions.append(categories.c.name == filt.category)
    if filt.account is not None:
        conditions.append(accounts.c.name == filt.account)
    if filt.merchant is not None:
        conditions.append(merchants.c.name == filt.merchant)
    if filt.min_amount is not None:
        conditions.append(transactions.c.amount >= filt.min_amount)
    if filt.max_amount is not None:
        conditions.append(transactions.c.amount <= filt.max_amount)
    return conditions


def get_transactions(
    engine: Engine, filt: TransactionFilter, limit: int = 50
) -> list[TransactionRow]:
    """Return up to `limit` raw transactions matching `filt`, most recent first."""
    if filt.category is not None:
        validate_category(engine, filt.category)

    query = (
        joined_transactions()
        .where(*filter_conditions(filt))
        .order_by(transactions.c.date.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(query).all()
    return [TransactionRow.model_validate(row._mapping) for row in rows]


def aggregate_spending(
    engine: Engine,
    filt: TransactionFilter,
    group_by: Literal["none", "category", "month"] = "none",
) -> AggregateResult | list[AggregateResult]:
    """Sum and count transactions matching `filt`, optionally grouped by category or month.

    Returns a single `AggregateResult` for `group_by="none"`, or a list of one
    `AggregateResult` per group otherwise.
    """
    if filt.category is not None:
        validate_category(engine, filt.category)

    query = joined_transactions().where(*filter_conditions(filt))
    with engine.connect() as conn:
        rows = conn.execute(query).all()

    if group_by == "none":
        total = sum((row.amount for row in rows), Decimal("0"))
        return AggregateResult(total=total, count=len(rows), group_key=None, filter_applied=filt)

    groups: dict[str, list[Decimal]] = defaultdict(list)
    for row in rows:
        key = row.category if group_by == "category" else row.date.strftime("%Y-%m")
        groups[key].append(row.amount)

    return [
        AggregateResult(
            total=sum(amounts, Decimal("0")),
            count=len(amounts),
            group_key=key,
            filter_applied=filt,
        )
        for key, amounts in sorted(groups.items())
    ]
