"""The LangGraph state schema for the agent's turn loop.

This is the minimal-ReAct-slice version of the state: just enough to run the
act/tool_node/draft_answer loop for a single-turn question. Routing, groundedness,
and cross-turn memory fields are added in later slices per the LLD's incremental
build plan.
"""

from typing import Any, TypedDict

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


def initial_state(question: str) -> AgentState:
    """Build the starting state for a single-turn run of the graph."""
    return AgentState(
        question=question,
        messages=[SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        ledger=[],
        draft_answer=None,
    )
