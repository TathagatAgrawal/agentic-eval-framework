"""The LangGraph state schema for the agent's turn loop.

This slice adds `resolved_question`, `is_ambiguous`, `ambiguity_reason`, and
`session_memory` on top of the groundedness slice, so a follow-up question can
be resolved against prior turns before routing/acting. The graph itself stays a
pure function of `AgentState` -- it reads `session_memory` as input but does not
mutate an external session object; the caller (the CLI's chat loop, in a later
phase) is responsible for turning the returned state into a `TurnMemory` and
appending it to its own `SessionMemory` between turns. Because `messages` starts
fresh each turn (no HumanMessage until `contextualize` adds the resolved one),
cross-turn continuity flows entirely through `session_memory`, not raw chat
history -- this is what keeps reference resolution grounded in something
unambiguous rather than free text, per the LLD's conversational-memory design.
"""

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage

from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.prompts import SYSTEM_PROMPT


class LedgerEntry(TypedDict):
    """One tool call and its result, recorded so the final answer can cite it and
    so groundedness can be checked structurally."""

    ledger_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any  # a tool's serialized result: a dict, or a list for grouped/multi-row tools
    timestamp: str


class TurnMemory(TypedDict):
    """A completed turn's structured summary, kept across turns in a session so
    follow-up questions can resolve against what was actually established
    (not just what was said)."""

    turn_id: int
    raw_question: str
    resolved_question: str
    final_answer: str
    ledger: list[LedgerEntry]


class AgentState(TypedDict):
    """The full state threaded through the agent's LangGraph nodes for one turn."""

    question: str
    resolved_question: str | None
    is_ambiguous: bool
    ambiguity_reason: str | None
    session_memory: list[TurnMemory]
    messages: list[BaseMessage]
    ledger: list[LedgerEntry]
    draft_answer: StructuredAnswer | None
    route: Literal["answer", "clarify", "refuse"] | None
    route_reason: str | None
    groundedness_ok: bool
    retry_count: int
    final_answer: str | None


def initial_state(question: str, session_memory: list[TurnMemory] | None = None) -> AgentState:
    """Build the starting state for one turn, optionally carrying prior turns'
    `TurnMemory` for follow-up reference resolution."""
    return AgentState(
        question=question,
        resolved_question=None,
        is_ambiguous=False,
        ambiguity_reason=None,
        session_memory=session_memory or [],
        messages=[SystemMessage(content=SYSTEM_PROMPT)],
        ledger=[],
        draft_answer=None,
        route=None,
        route_reason=None,
        groundedness_ok=False,
        retry_count=0,
        final_answer=None,
    )


def turn_memory_from_state(state: AgentState) -> TurnMemory:
    """Build the `TurnMemory` a caller should append to its `SessionMemory` after
    a graph run completes, so the next turn can resolve follow-ups against it."""
    assert state["resolved_question"] is not None
    assert state["final_answer"] is not None
    return TurnMemory(
        turn_id=len(state["session_memory"]) + 1,
        raw_question=state["question"],
        resolved_question=state["resolved_question"],
        final_answer=state["final_answer"],
        ledger=state["ledger"],
    )
