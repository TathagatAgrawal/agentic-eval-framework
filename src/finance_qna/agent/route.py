"""The routing decision schema produced by the `route` node.

Classifying the question before any tool is called is what lets the agent
refuse out-of-scope questions and ask for clarification instead of guessing,
per the feature-list requirement to decline or clarify rather than fabricate.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Where a question should be routed, and why."""

    route: Literal["answer", "clarify", "refuse"] = Field(
        description=(
            "'answer' if the question can be answered from the user's transaction data "
            "and is unambiguous; 'clarify' if a time period or reference is genuinely "
            "ambiguous and shouldn't be guessed; 'refuse' if the question isn't about "
            "the user's own transaction data at all."
        )
    )
    reason: str = Field(description="A short explanation of why this route was chosen.")
