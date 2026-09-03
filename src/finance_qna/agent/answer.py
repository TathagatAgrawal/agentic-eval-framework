"""The structured final-answer schema the draft_answer node must produce.

Requiring every numeric value in the answer to cite a `ledger_id` (rather than
letting the model emit free-form prose) is what makes the groundedness check in
`groundedness.py` a deterministic structural check instead of NLP parsing.
"""

from decimal import Decimal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One numeric value stated in the answer, and the ledger entry it came from."""

    value: Decimal = Field(description="The exact numeric value as stated in the answer text.")
    ledger_id: str = Field(
        description="The ledger_id of the tool result this value came from (or was derived from)."
    )
    computation: str | None = Field(
        default=None,
        description=(
            "If this value isn't a raw ledger value, a short description of how it was "
            "derived (e.g. 'sum of L1 and L2')."
        ),
    )


class StructuredAnswer(BaseModel):
    """The agent's answer, with every numeric claim traced to a ledger entry."""

    text: str = Field(description="The natural-language answer to show the user.")
    claims: list[Claim] = Field(
        default_factory=list,
        description="Every numeric value that appears in `text`, each citing its source.",
    )
