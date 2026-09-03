"""The output schema for the `contextualize` node.

Resolving a follow-up into a self-contained question before routing/acting is
what lets implicit references ("and what about April?"), scope carry-over
("what about just groceries?"), and comparative follow-ups ("is that more than
last year?") work without the agent guessing -- per the feature-list requirement
to maintain conversational memory across turns.
"""

from pydantic import BaseModel, Field


class ContextualizeResult(BaseModel):
    """The resolved, self-contained question, or a flag that it can't be resolved."""

    resolved_question: str = Field(
        description=(
            "The question rewritten to be fully self-contained, resolving any "
            "implicit reference, scope carry-over, or comparison against a prior "
            "turn using the conversation history provided. If the question is "
            "already self-contained, repeat it unchanged. If it's ambiguous, put "
            "your best-effort interpretation here anyway."
        )
    )
    is_ambiguous: bool = Field(
        description=(
            "True only if the question depends on a reference (a time period, "
            "category, or prior answer) that genuinely cannot be resolved from "
            "the conversation history -- not just because it's a follow-up, but "
            "because there's no reasonable antecedent for it."
        )
    )
    ambiguity_reason: str | None = Field(
        default=None,
        description="If is_ambiguous, a short explanation of what's unresolvable.",
    )
