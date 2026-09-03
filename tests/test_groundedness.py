"""Tests for the pure, LLM-free groundedness verifier."""

from decimal import Decimal

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.groundedness import verify
from finance_qna.agent.state import LedgerEntry


def _ledger_entry(ledger_id: str, result: dict | list) -> LedgerEntry:
    """Build a minimal LedgerEntry for a given result payload."""
    return LedgerEntry(
        ledger_id=ledger_id,
        tool_name="aggregate_spending_tool",
        args={},
        result=result,
        timestamp="2025-01-01T00:00:00+00:00",
    )


def test_verify_passes_claim_matching_a_raw_ledger_value() -> None:
    """A claim whose value equals a ledger result field must pass."""
    ledger = [_ledger_entry("L1", {"total": "415.77", "count": 8})]
    draft = StructuredAnswer(
        text="You spent $415.77.", claims=[Claim(value=Decimal("415.77"), ledger_id="L1")]
    )

    result = verify(draft, ledger)

    assert result.ok
    assert result.unsupported == []


def test_verify_fails_claim_with_no_matching_value() -> None:
    """A claim citing a value that doesn't appear anywhere in the ledger must fail."""
    ledger = [_ledger_entry("L1", {"total": "415.77", "count": 8})]
    draft = StructuredAnswer(
        text="You spent $999.99.", claims=[Claim(value=Decimal("999.99"), ledger_id="L1")]
    )

    result = verify(draft, ledger)

    assert not result.ok
    assert result.unsupported == draft.claims


def test_verify_passes_claim_derived_as_a_sum_of_two_ledger_values() -> None:
    """A claim equal to the sum of two different ledger entries' totals must pass."""
    ledger = [
        _ledger_entry("L1", {"total": "100.00"}),
        _ledger_entry("L2", {"total": "50.00"}),
    ]
    draft = StructuredAnswer(
        text="Combined, that's $150.00.",
        claims=[Claim(value=Decimal("150.00"), ledger_id="L1", computation="sum of L1 and L2")],
    )

    result = verify(draft, ledger)

    assert result.ok


def test_verify_passes_claim_derived_as_a_percentage_change() -> None:
    """A claim equal to the percentage change between two ledger values must pass."""
    ledger = [
        _ledger_entry("L1", {"total": "100.00"}),
        _ledger_entry("L2", {"total": "110.00"}),
    ]
    draft = StructuredAnswer(
        text="That's a 10% increase.",
        claims=[Claim(value=Decimal("10"), ledger_id="L1", computation="pct change L1 to L2")],
    )

    result = verify(draft, ledger)

    assert result.ok


def test_verify_matches_within_tolerance() -> None:
    """A claim within the tolerance of a ledger value must still pass."""
    ledger = [_ledger_entry("L1", {"total": "415.77"})]
    draft = StructuredAnswer(
        text="Approximately $415.77.", claims=[Claim(value=Decimal("415.775"), ledger_id="L1")]
    )

    result = verify(draft, ledger, tol=Decimal("0.01"))

    assert result.ok


def test_verify_passes_with_no_claims() -> None:
    """An answer with no numeric claims trivially passes (nothing to verify)."""
    draft = StructuredAnswer(text="No spending found.", claims=[])

    result = verify(draft, ledger=[])

    assert result.ok


def test_verify_finds_values_nested_in_list_results() -> None:
    """Values inside a grouped (list) tool result must still be checked."""
    ledger = [
        _ledger_entry(
            "L1",
            [
                {"total": "200.00", "group_key": "Dining"},
                {"total": "300.00", "group_key": "Groceries"},
            ],
        )
    ]
    draft = StructuredAnswer(
        text="You spent $300.00 on groceries.",
        claims=[Claim(value=Decimal("300.00"), ledger_id="L1")],
    )

    result = verify(draft, ledger)

    assert result.ok
