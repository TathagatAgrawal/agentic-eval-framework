"""Deterministic, LLM-free verification that every claimed number is grounded.

This is what turns "every numeric claim traces back to a tool result" from a
prompting hope into a structural guarantee: a claim passes only if its value
matches a number that actually appears somewhere in this turn's ledger, or a
simple sum/difference/percentage derived from two ledger values.
"""

from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.state import LedgerEntry

DEFAULT_TOLERANCE = Decimal("0.01")


class GroundCheckResult(BaseModel):
    """The outcome of checking a draft answer's claims against the ledger."""

    ok: bool
    unsupported: list[Claim]


def _extract_decimals(value: object) -> list[Decimal]:
    """Recursively collect every value in a JSON-like structure that parses as
    a Decimal (numbers and numeric strings), skipping booleans."""
    found: list[Decimal] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, int | float):
        found.append(Decimal(str(value)))
    elif isinstance(value, str):
        try:
            found.append(Decimal(value))
        except InvalidOperation:
            pass
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(_extract_decimals(v))
    elif isinstance(value, list):
        for item in value:
            found.extend(_extract_decimals(item))
    return found


def _matches_derived_value(value: Decimal, all_values: list[Decimal], tol: Decimal) -> bool:
    """Check whether `value` is the sum, difference, or percentage change of any
    two values across the whole ledger (covers "is that more or less" claims)."""
    for a in all_values:
        for b in all_values:
            if a is b:
                continue
            if abs((a + b) - value) <= tol:
                return True
            if abs((a - b) - value) <= tol:
                return True
            if a != 0 and abs(((b - a) / a * Decimal("100")) - value) <= tol:
                return True
    return False


def verify(
    draft: StructuredAnswer,
    ledger: list[LedgerEntry],
    tol: Decimal = DEFAULT_TOLERANCE,
) -> GroundCheckResult:
    """Check every claim in `draft` against `ledger`, returning which (if any)
    claims aren't supported by a ledger value or a simple derived combination."""
    values_by_ledger_id: dict[str, list[Decimal]] = {
        entry["ledger_id"]: _extract_decimals(entry["result"]) for entry in ledger
    }
    all_values = [v for values in values_by_ledger_id.values() for v in values]

    unsupported: list[Claim] = []
    for claim in draft.claims:
        candidates = values_by_ledger_id.get(claim.ledger_id, [])
        if any(abs(v - claim.value) <= tol for v in candidates):
            continue
        if _matches_derived_value(claim.value, all_values, tol):
            continue
        unsupported.append(claim)

    return GroundCheckResult(ok=not unsupported, unsupported=unsupported)
