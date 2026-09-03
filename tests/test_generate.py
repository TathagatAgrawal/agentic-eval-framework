"""Tests for the synthetic data generator: determinism and labeled ground truth."""

import json
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from finance_qna.data.db import get_engine
from finance_qna.data.generate import generate
from finance_qna.data.schema import merchants, transactions


def test_generate_is_deterministic(tmp_path: Path) -> None:
    """Two runs with the same seed must produce byte-identical DB and ground truth."""
    db_a = tmp_path / "a.db"
    gt_a = tmp_path / "a_ground_truth.json"
    db_b = tmp_path / "b.db"
    gt_b = tmp_path / "b_ground_truth.json"

    generate(seed=42, db_path=db_a, ground_truth_path=gt_a)
    generate(seed=42, db_path=db_b, ground_truth_path=gt_b)

    assert gt_a.read_bytes() == gt_b.read_bytes()
    assert db_a.read_bytes() == db_b.read_bytes()


def test_generate_produces_rows_and_matching_ground_truth(tmp_path: Path) -> None:
    """The row count in the database must match the count recorded in ground_truth.json."""
    db_path = tmp_path / "test.db"
    gt_path = tmp_path / "ground_truth.json"
    generate(seed=42, db_path=db_path, ground_truth_path=gt_path)

    ground_truth = json.loads(gt_path.read_text())
    assert ground_truth["transaction_count"] > 0

    engine = get_engine(db_path)
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(transactions)).scalar_one()
        assert count == ground_truth["transaction_count"]


def test_subscription_price_change_is_labeled_and_present(tmp_path: Path) -> None:
    """The labeled subscription's transactions must switch from old to new amount
    exactly at the labeled change month."""
    db_path = tmp_path / "test.db"
    gt_path = tmp_path / "ground_truth.json"
    generate(seed=42, db_path=db_path, ground_truth_path=gt_path)

    ground_truth = json.loads(gt_path.read_text())
    event = ground_truth["subscription_change_event"]

    engine = get_engine(db_path)
    with engine.connect() as conn:
        rows = conn.execute(
            select(transactions.c.date, transactions.c.amount)
            .select_from(transactions.join(merchants))
            .where(merchants.c.name == event["merchant"])
            .order_by(transactions.c.date)
        ).all()

    before = [r.amount for r in rows if r.date.isoformat() < event["change_month"]]
    after = [r.amount for r in rows if r.date.isoformat() >= event["change_month"]]

    assert before, "expected transactions before the change month"
    assert after, "expected transactions after the change month"
    assert all(a == Decimal(event["old_amount"]) for a in before)
    assert all(a == Decimal(event["new_amount"]) for a in after)


def test_spike_event_is_labeled_and_reflected_in_totals(tmp_path: Path) -> None:
    """The labeled spike month's category total must exceed every other month's total
    for that category."""
    db_path = tmp_path / "test.db"
    gt_path = tmp_path / "ground_truth.json"
    generate(seed=42, db_path=db_path, ground_truth_path=gt_path)

    ground_truth = json.loads(gt_path.read_text())
    spike = ground_truth["spike_event"]
    monthly_totals = ground_truth["monthly_category_totals"][spike["category"]]
    other_months = [
        Decimal(total) for month, total in monthly_totals.items() if month != spike["month"]
    ]
    spike_month_total = Decimal(monthly_totals[spike["month"]])

    assert spike_month_total > max(other_months)
