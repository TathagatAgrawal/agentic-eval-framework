"""Assembles the agent's LangGraph state machine (routing + ReAct slice).

Graph shape for this slice: route classifies the question first; "clarify" and
"refuse" go straight to a short response with no tool calls; "answer" enters the
act/tool_node loop until it stops requesting tools or the step limit is hit, then
draft_answer produces the final, ledger-cited answer. Groundedness checking and
multi-turn memory are added on top of this in the next slice per the LLD's
incremental build plan.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from sqlalchemy import Engine

from finance_qna.agent.nodes import (
    make_act_node,
    make_clarify_node,
    make_draft_answer_node,
    make_refuse_node,
    make_route_node,
    make_router,
    make_tool_node,
    route_edge,
)
from finance_qna.agent.state import AgentState
from finance_qna.tools.registry import build_tools


def build_graph(engine: Engine, llm: BaseChatModel, max_tool_steps: int = 6) -> Any:
    """Compile the agent graph, wired to `engine`'s data and `llm` for reasoning."""
    tools = build_tools(engine)

    graph = StateGraph(AgentState)
    graph.add_node("route", make_route_node(llm))
    graph.add_node("clarify", make_clarify_node(llm))
    graph.add_node("refuse", make_refuse_node(llm))
    graph.add_node("act", make_act_node(llm, tools))
    graph.add_node("tool_node", make_tool_node(tools))
    graph.add_node("draft_answer", make_draft_answer_node(llm))

    graph.add_edge(START, "route")
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
    graph.add_edge("draft_answer", END)

    return graph.compile()
