"""Assembles the agent's LangGraph state machine (contextualize + routing + ReAct
+ groundedness -- the full Phase 4 graph).

Graph shape: contextualize resolves the question against prior turns first
(flagging it as ambiguous if it can't be); route then classifies it as
answer/clarify/refuse, short-circuiting straight to "clarify" if contextualize
already flagged ambiguity. "clarify" and "refuse" go straight to a short
response with no tool calls. "answer" enters the act/tool_node loop until it
stops requesting tools or the step limit is hit, then draft_answer produces a
candidate answer, which ground_check verifies against the ledger -- accepting
it, sending it back to act for a bounded retry with a note about what was
unsupported, or falling back to a caveated response once retries are exhausted.

The graph itself is a pure function of `AgentState`: it reads `session_memory`
as input but never mutates an external session object. Turning a completed run
into the next turn's `TurnMemory` (via `state.turn_memory_from_state`) and
appending it to a `SessionMemory` is the caller's job -- see `memory/session.py`.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from sqlalchemy import Engine

from finance_qna.agent.nodes import (
    make_act_node,
    make_clarify_node,
    make_contextualize_node,
    make_draft_answer_node,
    make_ground_check_node,
    make_ground_router,
    make_refuse_node,
    make_respond_node,
    make_respond_with_caveat_node,
    make_route_node,
    make_router,
    make_tool_node,
    route_edge,
)
from finance_qna.agent.state import AgentState
from finance_qna.tools.registry import build_tools


def build_graph(
    engine: Engine,
    llm: BaseChatModel,
    max_tool_steps: int = 6,
    groundedness_retry_limit: int = 1,
) -> Any:
    """Compile the agent graph, wired to `engine`'s data and `llm` for reasoning."""
    tools = build_tools(engine)

    graph = StateGraph(AgentState)
    graph.add_node("contextualize", make_contextualize_node(llm))
    graph.add_node("route", make_route_node(llm))
    graph.add_node("clarify", make_clarify_node(llm))
    graph.add_node("refuse", make_refuse_node(llm))
    graph.add_node("act", make_act_node(llm, tools))
    graph.add_node("tool_node", make_tool_node(tools))
    graph.add_node("draft_answer", make_draft_answer_node(llm))
    graph.add_node("ground_check", make_ground_check_node())
    graph.add_node("respond", make_respond_node())
    graph.add_node("respond_with_caveat", make_respond_with_caveat_node())

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "route")
    graph.add_conditional_edges(
        "route",
        route_edge,
        {"answer": "act", "clarify": "clarify", "refuse": "refuse"},
    )
    graph.add_edge("clarify", END)
    graph.add_edge("refuse", END)
    graph.add_conditional_edges(
        "act",
        make_router(max_tool_steps),
        {"tool_node": "tool_node", "draft_answer": "draft_answer"},
    )
    graph.add_edge("tool_node", "act")
    graph.add_edge("draft_answer", "ground_check")
    graph.add_conditional_edges(
        "ground_check",
        make_ground_router(groundedness_retry_limit),
        {"respond": "respond", "act": "act", "respond_with_caveat": "respond_with_caveat"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("respond_with_caveat", END)

    return graph.compile()
