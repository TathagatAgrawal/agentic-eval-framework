"""Tests for the eval harness's TestCase schema and the checked-in test set.

These validate the dataset itself (schema conformance, no duplicate ids, and a
handful of labels spot-checked against data/ground_truth.json) -- this is what
catches a typo in the test set before it ever reaches the runner.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval.schema import ExpectedResult, TestCase, load_test_cases

TESTSET_DIR = Path(__file__).parent.parent / "eval" / "testset"


def test_loads_exactly_fifteen_cases() -> None:
    """The curated v1 test set is deliberately capped at 15 cases."""
    cases = load_test_cases(TESTSET_DIR / "single_turn.yaml", TESTSET_DIR / "multi_turn.yaml")
    assert len(cases) == 15


def test_all_case_ids_are_unique() -> None:
    """load_test_cases must reject a duplicate id across files."""
    cases = load_test_cases(TESTSET_DIR / "single_turn.yaml", TESTSET_DIR / "multi_turn.yaml")
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_consistent_shape() -> None:
    """Every loaded case must be exactly single-turn or exactly multi-turn."""
    cases = load_test_cases(TESTSET_DIR / "single_turn.yaml", TESTSET_DIR / "multi_turn.yaml")
    for case in cases:
        if case.type == "single_turn":
            assert case.question is not None
            assert case.expected is not None
            assert case.turns is None
        else:
            assert case.turns is not None
            assert len(case.turns) > 0


def test_answer_cases_declare_a_target_value() -> None:
    """Every 'answer' expectation (at the case or turn level) must have a value to check."""
    cases = load_test_cases(TESTSET_DIR / "single_turn.yaml", TESTSET_DIR / "multi_turn.yaml")
    for case in cases:
        expecteds = (
            [case.expected]
            if case.type == "single_turn"
            else [t.expected for t in case.turns or []]
        )
        for expected in expecteds:
            assert expected is not None
            if expected.behavior == "answer":
                assert expected.value is not None or expected.values is not None


def test_expected_result_rejects_answer_with_no_value() -> None:
    """The schema itself must reject an 'answer' expectation with no value/values."""
    with pytest.raises(ValidationError):
        ExpectedResult(behavior="answer")


def test_test_case_rejects_both_question_and_turns() -> None:
    """A test case can't be single-turn and multi-turn at once."""
    with pytest.raises(ValidationError):
        TestCase(
            id="bad",
            category="x",
            question="q",
            expected=ExpectedResult(behavior="refuse"),
            turns=[{"question": "q2", "expected": {"behavior": "refuse"}}],
        )


def test_test_case_rejects_neither_question_nor_turns() -> None:
    """A test case must be either single-turn or multi-turn."""
    with pytest.raises(ValidationError):
        TestCase(id="bad", category="x")


def test_simple_lookup_values_match_ground_truth(ground_truth: dict[str, Any]) -> None:
    """Spot-check the simple-lookup labels directly against ground_truth.json,
    so a typo in the YAML can't silently diverge from the data it's meant to check."""
    cases = load_test_cases(TESTSET_DIR / "single_turn.yaml")
    by_id = {case.id: case for case in cases}

    groceries_july_2024 = Decimal(
        ground_truth["monthly_category_totals"]["Groceries"]["2024-07-01"]
    )
    assert by_id["simple_001"].expected is not None
    assert Decimal(str(by_id["simple_001"].expected.value)) == groceries_july_2024

    dining_dec_2025 = Decimal(ground_truth["monthly_category_totals"]["Dining"]["2025-12-01"])
    assert by_id["simple_002"].expected is not None
    assert Decimal(str(by_id["simple_002"].expected.value)) == dining_dec_2025


def test_trend_case_matches_subscription_change_event(ground_truth: dict[str, Any]) -> None:
    """trend_001's expected value must match the generator's labeled subscription event."""
    cases = load_test_cases(TESTSET_DIR / "single_turn.yaml")
    by_id = {case.id: case for case in cases}

    expected_new_amount = Decimal(ground_truth["subscription_change_event"]["new_amount"])
    assert by_id["trend_001"].expected is not None
    assert Decimal(str(by_id["trend_001"].expected.value)) == expected_new_amount
