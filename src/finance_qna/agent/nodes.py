"""LangGraph node functions for the agent's turn loop (routing + ReAct slice).

Each `make_*_node` factory closes over its dependencies (the LLM, the tool list,
the step limit) and returns a plain `state -> partial state` function, so the
nodes themselves stay free of global state and are easy to unit test in
isolation with a hand-built `AgentState`.
"""

import json
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.prompts import (
    CLARIFY_INSTRUCTIONS,
    DRAFT_ANSWER_INSTRUCTIONS,
    REFUSE_INSTRUCTIONS,
    ROUTE_INSTRUCTIONS,
)
from finance_qna.agent.route import RouteDecision
from finance_qna.agent.state import AgentState, LedgerEntry
from finance_qna.tools.errors import CategoryNotFoundError


def serialize_tool_result(result: Any) -> Any:
    """Convert a tool's return value (a Pydantic model, a list of them, or a
    plain value) into a JSON-serializable structure."""
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, list):
        return [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in result
        ]
    return result


def extract_text(content: Any) -> str:
    """Extract plain text from an AIMessage's `content`.

    Some providers (including the current Gemini integration) return `content`
    as a list of content blocks (e.g. `[{"type": "text", "text": "..."}]`)
    rather than a plain string, so this normalizes either shape to text.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def make_act_node(llm: BaseChatModel, tools: list[BaseTool]) -> Any:
    """Build the `act` node: proposes a tool call, or stops when it has enough
    information to answer."""
    llm_with_tools = llm.bind_tools(tools)

    def act(state: AgentState) -> dict[str, Any]:
        """Run one reasoning step over the current conversation and tool history."""
        response = llm_with_tools.invoke(state["messages"])
        assert isinstance(response, AIMessage)
        return {"messages": [*state["messages"], response]}

    return act


def make_tool_node(tools: list[BaseTool]) -> Any:
    """Build the tool-execution node: runs every requested tool call, appends a
    ledger entry for each successful one, and reports errors back to the model."""
    tools_by_name = {t.name: t for t in tools}

    def tool_node(state: AgentState) -> dict[str, Any]:
        """Execute the pending tool calls from the last AI message."""
        last_message = state["messages"][-1]
        assert isinstance(last_message, AIMessage)

        tool_messages: list[ToolMessage] = []
        new_ledger: list[LedgerEntry] = list(state["ledger"])

        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            try:
                raw_result = tool.invoke(tool_call["args"])
            except CategoryNotFoundError as exc:
                content = json.dumps({"error": str(exc), "valid_categories": exc.valid_categories})
            except ValidationError as exc:
                content = json.dumps({"error": f"invalid arguments: {exc}"})
            else:
                ledger_id = f"L{len(new_ledger) + 1}"
                serialized = serialize_tool_result(raw_result)
                new_ledger.append(
                    LedgerEntry(
                        ledger_id=ledger_id,
                        tool_name=tool_call["name"],
                        args=tool_call["args"],
                        result=serialized,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
                content = json.dumps({"ledger_id": ledger_id, "result": serialized})

            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

        return {
            "messages": [*state["messages"], *tool_messages],
            "ledger": new_ledger,
        }

    return tool_node


def make_router(max_tool_steps: int) -> Any:
    """Build the conditional-edge function deciding whether to keep calling tools."""

    def router(state: AgentState) -> Literal["tool_node", "draft_answer"]:
        """Route to another tool call, or to drafting the final answer."""
        last_message = state["messages"][-1]
        has_pending_tool_calls = (
            isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0
        )
        if has_pending_tool_calls and len(state["ledger"]) < max_tool_steps:
            return "tool_node"
        return "draft_answer"

    return router


def make_draft_answer_node(llm: BaseChatModel) -> Any:
    """Build the `draft_answer` node: produces the final, ledger-cited answer."""
    structured_llm = llm.with_structured_output(StructuredAnswer)

    def draft_answer(state: AgentState) -> dict[str, Any]:
        """Compose the final answer from the question and this turn's ledger."""
        ledger_text = (
            "\n".join(
                f"{entry['ledger_id']} ({entry['tool_name']}): {json.dumps(entry['result'])}"
                for entry in state["ledger"]
            )
            or "(no tool calls were made)"
        )

        prompt = DRAFT_ANSWER_INSTRUCTIONS.format(question=state["question"], ledger=ledger_text)
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        assert isinstance(result, StructuredAnswer)
        return {"draft_answer": result}

    return draft_answer


def make_route_node(llm: BaseChatModel) -> Any:
    """Build the `route` node: classifies the question as answer/clarify/refuse
    before any tool is called."""
    structured_llm = llm.with_structured_output(RouteDecision)

    def route(state: AgentState) -> dict[str, Any]:
        """Classify `state['question']` and record the route and its reason."""
        prompt = ROUTE_INSTRUCTIONS.format(question=state["question"])
        decision = structured_llm.invoke([HumanMessage(content=prompt)])
        assert isinstance(decision, RouteDecision)
        return {"route": decision.route, "route_reason": decision.reason}

    return route


def route_edge(state: AgentState) -> Literal["answer", "clarify", "refuse"]:
    """Conditional-edge function: send the graph down the branch `route` chose."""
    assert state["route"] is not None
    return state["route"]


def make_clarify_node(llm: BaseChatModel) -> Any:
    """Build the `clarify` node: asks a short clarifying question instead of guessing."""

    def clarify(state: AgentState) -> dict[str, Any]:
        """Generate a clarifying question and store it as the turn's answer."""
        prompt = CLARIFY_INSTRUCTIONS.format(
            question=state["question"], reason=state["route_reason"] or ""
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_answer": StructuredAnswer(text=extract_text(response.content), claims=[])}

    return clarify


def make_refuse_node(llm: BaseChatModel) -> Any:
    """Build the `refuse` node: politely declines an out-of-scope question."""

    def refuse(state: AgentState) -> dict[str, Any]:
        """Generate a decline message and store it as the turn's answer."""
        prompt = REFUSE_INSTRUCTIONS.format(
            question=state["question"], reason=state["route_reason"] or ""
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_answer": StructuredAnswer(text=extract_text(response.content), claims=[])}

    return refuse
