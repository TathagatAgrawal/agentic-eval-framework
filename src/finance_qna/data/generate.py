"""Deterministic synthetic transaction data generator.

Generates a portfolio-safe synthetic dataset anchored to a fixed reference date
(not `datetime.now()`), so re-running with the same seed always produces the same
database and the same `ground_truth.json` sidecar -- the eval harness's labels are
derived from this same generation run, so they can never drift from the data.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta
from faker import Faker
from sqlalchemy import CursorResult, insert

from finance_qna.data.db import get_engine
from finance_qna.data.schema import accounts, categories, merchants, transactions

# The synthetic dataset's "today" -- kept fixed so the data, and therefore every
# ground-truth aggregate, is reproducible across runs and across time.
REFERENCE_DATE = date(2025, 12, 31)
NUM_MONTHS = 18  # generates transactions from REFERENCE_DATE - 18mo .. REFERENCE_DATE

ACCOUNTS = [
    {"name": "Checking", "type": "checking"},
    {"name": "Credit Card", "type": "credit_card"},
]

# (name, parent_name | None)
CATEGORIES: list[tuple[str, str | None]] = [
    ("Food", None),
    ("Dining", "Food"),
    ("Groceries", "Food"),
    ("Transport", None),
    ("Housing", None),
    ("Utilities", None),
    ("Entertainment", None),
    ("Subscriptions", None),
    ("Shopping", None),
    ("Health", None),
    ("Income", None),
]

# category -> [(merchant_name, is_subscription, min_amount, max_amount, txns_per_month)]
MERCHANT_SPEC: dict[str, list[tuple[str, bool, Decimal, Decimal, float]]] = {
    "Dining": [
        ("The Corner Bistro", False, Decimal("12.00"), Decimal("55.00"), 8),
        ("Sushi Place", False, Decimal("18.00"), Decimal("60.00"), 3),
        ("Coffee Shop", False, Decimal("4.00"), Decimal("9.00"), 10),
    ],
    "Groceries": [
        ("Fresh Market", False, Decimal("30.00"), Decimal("140.00"), 5),
        ("Corner Grocer", False, Decimal("10.00"), Decimal("45.00"), 4),
    ],
    "Transport": [
        ("Metro Transit", False, Decimal("2.50"), Decimal("2.50"), 12),
        ("RideShare Co", False, Decimal("8.00"), Decimal("35.00"), 6),
        ("Gas Station", False, Decimal("30.00"), Decimal("60.00"), 3),
    ],
    "Housing": [
        ("Maple Ave Apartments", False, Decimal("1500.00"), Decimal("1500.00"), 1),
    ],
    "Utilities": [
        ("City Power & Light", False, Decimal("60.00"), Decimal("140.00"), 1),
        ("Metro Water", False, Decimal("25.00"), Decimal("45.00"), 1),
        ("FastNet Internet", False, Decimal("55.00"), Decimal("55.00"), 1),
    ],
    "Entertainment": [
        ("Cineplex", False, Decimal("12.00"), Decimal("40.00"), 2),
        ("Concert Hall", False, Decimal("40.00"), Decimal("120.00"), 0.3),
    ],
    "Subscriptions": [
        ("StreamFlix", True, Decimal("13.99"), Decimal("13.99"), 1),
        ("GymPass", True, Decimal("29.99"), Decimal("29.99"), 1),
        ("CloudDrive+", True, Decimal("9.99"), Decimal("9.99"), 1),
    ],
    "Shopping": [
        ("MegaMart", False, Decimal("15.00"), Decimal("120.00"), 4),
        ("Online Retailer", False, Decimal("10.00"), Decimal("200.00"), 5),
    ],
    "Health": [
        ("City Pharmacy", False, Decimal("8.00"), Decimal("45.00"), 2),
        ("Family Clinic", False, Decimal("25.00"), Decimal("150.00"), 0.5),
    ],
    "Income": [
        ("Employer Payroll", False, Decimal("-3200.00"), Decimal("-3200.00"), 2),
    ],
}

# The subscription merchant + month whose price changes -- the labeled ground
# truth for `detect_subscription_changes`.
SUBSCRIPTION_CHANGE_MERCHANT = "StreamFlix"
SUBSCRIPTION_CHANGE_MONTH = date(2025, 6, 1)
SUBSCRIPTION_OLD_AMOUNT = Decimal("13.99")
SUBSCRIPTION_NEW_AMOUNT = Decimal("17.99")

# The category + month with an injected spend spike -- the labeled ground truth
# for anomaly-detection test cases.
SPIKE_CATEGORY = "Shopping"
SPIKE_MONTH = date(2025, 11, 1)
SPIKE_EXTRA_AMOUNT = Decimal("650.00")


def _quantize(amount: Decimal) -> Decimal:
    """Round a Decimal amount to 2 decimal places (currency precision), half-up."""
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _inserted_id(result: CursorResult[Any]) -> int:
    """Extract the auto-generated integer primary key from a single-row insert result."""
    pk = result.inserted_primary_key
    assert pk is not None
    return int(pk[0])


def _month_starts(reference_date: date, num_months: int) -> list[date]:
    """Return the first-of-month date for each of the `num_months` months ending in
    `reference_date`'s month, oldest first."""
    first_month = (reference_date.replace(day=1)) - relativedelta(months=num_months - 1)
    return [first_month + relativedelta(months=i) for i in range(num_months)]


@dataclass
class GeneratedTransaction:
    """A single synthetic transaction before it has been written to the database."""

    account_name: str
    merchant_name: str
    category_name: str
    date: date
    amount: Decimal
    description: str
    is_subscription: bool


@dataclass
class GenerationResult:
    """The full output of one generation run: transactions plus the ground-truth
    aggregates and labeled events derived from them."""

    transactions: list[GeneratedTransaction] = field(default_factory=list)
    monthly_category_totals: dict[str, dict[str, str]] = field(default_factory=dict)
    subscription_change_event: dict[str, str] = field(default_factory=dict)
    spike_event: dict[str, str] = field(default_factory=dict)


def _generate_transactions(seed: int) -> GenerationResult:
    """Generate the full set of synthetic transactions and their ground-truth
    aggregates for the given seed, without touching the database."""
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    result = GenerationResult()
    months = _month_starts(REFERENCE_DATE, NUM_MONTHS)
    account_names = [a["name"] for a in ACCOUNTS]

    monthly_totals: dict[tuple[str, date], Decimal] = {}

    for category_name, merchant_specs in MERCHANT_SPEC.items():
        for month_start in months:
            month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
            for merchant_name, is_sub, lo, hi, txns_per_month in merchant_specs:
                count = int(txns_per_month)
                if rng.random() < (txns_per_month - count):
                    count += 1

                for _ in range(count):
                    if is_sub:
                        amount = lo
                        if (
                            merchant_name == SUBSCRIPTION_CHANGE_MERCHANT
                            and month_start >= SUBSCRIPTION_CHANGE_MONTH
                        ):
                            amount = SUBSCRIPTION_NEW_AMOUNT
                        day = min(5, (month_end - month_start).days)
                    else:
                        amount = Decimal(str(round(rng.uniform(float(lo), float(hi)), 2)))
                        day = rng.randint(0, (month_end - month_start).days)

                    amount = _quantize(amount)
                    txn_date = month_start + relativedelta(days=day)
                    account_name = rng.choice(account_names)
                    txn = GeneratedTransaction(
                        account_name=account_name,
                        merchant_name=merchant_name,
                        category_name=category_name,
                        date=txn_date,
                        amount=amount,
                        description=f"{merchant_name} purchase",
                        is_subscription=is_sub,
                    )
                    result.transactions.append(txn)
                    key = (category_name, month_start)
                    monthly_totals[key] = monthly_totals.get(key, Decimal("0")) + amount

    # inject the labeled spend spike as a handful of extra Shopping transactions
    spike_merchant = MERCHANT_SPEC[SPIKE_CATEGORY][0][0]
    remaining = SPIKE_EXTRA_AMOUNT
    num_spike_txns = 3
    for i in range(num_spike_txns):
        chunk = (
            _quantize(remaining / (num_spike_txns - i))
            if i < num_spike_txns - 1
            else _quantize(remaining)
        )
        remaining -= chunk
        txn_date = SPIKE_MONTH + relativedelta(days=10 + i)
        txn = GeneratedTransaction(
            account_name=rng.choice(account_names),
            merchant_name=spike_merchant,
            category_name=SPIKE_CATEGORY,
            date=txn_date,
            amount=chunk,
            description=f"{spike_merchant} purchase",
            is_subscription=False,
        )
        result.transactions.append(txn)
        key = (SPIKE_CATEGORY, SPIKE_MONTH)
        monthly_totals[key] = monthly_totals.get(key, Decimal("0")) + chunk

    for (category_name, month_start), total in monthly_totals.items():
        result.monthly_category_totals.setdefault(category_name, {})[month_start.isoformat()] = str(
            total
        )

    result.subscription_change_event = {
        "merchant": SUBSCRIPTION_CHANGE_MERCHANT,
        "old_amount": str(SUBSCRIPTION_OLD_AMOUNT),
        "new_amount": str(SUBSCRIPTION_NEW_AMOUNT),
        "change_month": SUBSCRIPTION_CHANGE_MONTH.isoformat(),
    }
    result.spike_event = {
        "category": SPIKE_CATEGORY,
        "month": SPIKE_MONTH.isoformat(),
        "extra_amount": str(SPIKE_EXTRA_AMOUNT),
    }
    return result


def generate(seed: int, db_path: Path, ground_truth_path: Path | None = None) -> None:
    """Generate the synthetic dataset into `db_path` and write the ground-truth sidecar.

    Deterministic for a given `seed`: re-running with the same seed and an empty/absent
    `db_path` always produces the same rows and the same ground_truth.json.
    """
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine(db_path)
    result = _generate_transactions(seed)

    with engine.begin() as conn:
        account_ids: dict[str, int] = {}
        for acc in ACCOUNTS:
            res = conn.execute(insert(accounts).values(**acc))
            account_ids[acc["name"]] = _inserted_id(res)

        category_ids: dict[str, int] = {}
        # parents first, then children, so parent_category_id resolves
        for name, parent in sorted(CATEGORIES, key=lambda c: c[1] is not None):
            parent_id = category_ids.get(parent) if parent else None
            res = conn.execute(insert(categories).values(name=name, parent_category_id=parent_id))
            category_ids[name] = _inserted_id(res)

        merchant_ids: dict[str, int] = {}
        for category_name, specs in MERCHANT_SPEC.items():
            for merchant_name, _is_sub, _lo, _hi, _freq in specs:
                if merchant_name in merchant_ids:
                    continue
                res = conn.execute(
                    insert(merchants).values(
                        name=merchant_name,
                        default_category_id=category_ids[category_name],
                    )
                )
                merchant_ids[merchant_name] = _inserted_id(res)

        rows = [
            {
                "account_id": account_ids[t.account_name],
                "merchant_id": merchant_ids[t.merchant_name],
                "category_id": category_ids[t.category_name],
                "date": t.date,
                "amount": t.amount,
                "description": t.description,
                "is_subscription": t.is_subscription,
            }
            for t in result.transactions
        ]
        if rows:
            conn.execute(insert(transactions), rows)

    gt_path = ground_truth_path or db_path.parent / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "reference_date": REFERENCE_DATE.isoformat(),
                "num_months": NUM_MONTHS,
                "monthly_category_totals": result.monthly_category_totals,
                "subscription_change_event": result.subscription_change_event,
                "spike_event": result.spike_event,
                "transaction_count": len(result.transactions),
            },
            indent=2,
            sort_keys=True,
        )
    )
