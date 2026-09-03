"""The LangGraph state schema for the agent's turn loop.

This slice adds `groundedness_ok`, `retry_count`, and `final_answer` on top of
the routing-and-ReAct slice, so a draft answer is verified against the ledger
before it's shown to the user, with a bounded retry before falling back to a
caveated response. Cross-turn memory fields are added in the next slice per the
LLD's incremental build plan.
"""

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.prompts import SYSTEM_PROMPT


class LedgerEntry(TypedDict):
    """One tool call and its result, recorded so the final answer can cite it and
    so groundedness can be checked structurally."""

    ledger_id: str
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    timestamp: str


class AgentState(TypedDict):
    """The full state threaded through the agent's LangGraph nodes for one turn."""

    question: str
    messages: list[BaseMessage]
    ledger: list[LedgerEntry]
    draft_answer: StructuredAnswer | None
    route: Literal["answer", "clarify", "refuse"] | None
    route_reason: str | None
    groundedness_ok: bool
    retry_count: int
    final_answer: str | None


def initial_state(question: str) -> AgentState:
    """Build the starting state for a single-turn run of the graph."""
    return AgentState(
        question=question,
        messages=[SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        ledger=[],
        draft_answer=None,
        route=None,
        route_reason=None,
        groundedness_ok=False,
        retry_count=0,
        final_answer=None,
    )
