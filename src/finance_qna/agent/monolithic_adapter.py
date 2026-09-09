"""The Monolithic ReAct architecture: one system prompt, one tool-calling loop,
one final structured-output call that decides routing and drafts the answer
together -- no separate `contextualize`, `route`, or retry stages.

See design/alternative-architectures-plan.md §2 for the full rationale. This is
the second `AgentAdapter` implementation (after `langgraph_adapter.py`), built
to prove the non-LangGraph adapter pattern with the lowest-risk architecture
before attempting Plan-and-Execute or Context-Stuffing.
"""

import json
import time
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine

from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.agent.groundedness import verify
from finance_qna.agent.nodes import serialize_tool_result
from finance_qna.agent.prompts import (
    GROUNDEDNESS_CAVEAT,
    MONOLITHIC_FINAL_INSTRUCTIONS,
    MONOLITHIC_SYSTEM_PROMPT,
)
from finance_qna.agent.state import LedgerEntry
from finance_qna.tools.errors import CategoryNotFoundError
from finance_qna.tools.registry import build_tools
from finance_qna.tracing.trace import RunTrace


class MonolithicResult(BaseModel):
    """The agent's single final decision: how to route the question, and (if
    answering) the answer text with its cited claims."""

    route: Literal["answer", "clarify", "refuse"] = Field(
        description="'answer' if the ledger has enough to answer; 'clarify' if the "
        "question is genuinely ambiguous; 'refuse' if it isn't about the user's own "
        "transaction data."
    )
    text: str = Field(description="The response to show the user.")
    claims: list[Claim] = Field(
        default_factory=list,
        description="Every financial figure in `text`, cited to a ledger_id. "
        "Empty for clarify/refuse.",
    )


class MonolithicReactConfig(BaseSettings):
    """Configuration specific to the Monolithic ReAct architecture, loaded from
    `MONOLITHIC_*` environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MONOLITHIC_", extra="ignore")

    model: str = "gemini-3.5-flash-lite"
    max_tool_steps: int = 6


@lru_cache
def get_monolithic_config() -> MonolithicReactConfig:
    """Return the process-wide `MonolithicReactConfig` instance, loaded once and cached."""
    return MonolithicReactConfig()


def _format_history(prior_turns: list[RunTrace]) -> str:
    """Render prior turns as plain Q&A text for the prompt -- no structured
    resolution step, unlike the LangGraph architecture's `TurnMemory`."""
    if not prior_turns:
        return "(no prior turns)"
    return "\n".join(
        f"User: {trace.question}\nAssistant: {trace.final_answer}" for trace in prior_turns
    )


def _format_ledger(ledger: list[LedgerEntry]) -> str:
    """Render the ledger as text for the final-decision prompt."""
    if not ledger:
        return "(no tool calls were made)"
    return "\n".join(
        f"{entry['ledger_id']} ({entry['tool_name']}): {json.dumps(entry['result'])}"
        for entry in ledger
    )


class MonolithicReactAdapter:
    """Runs the Monolithic ReAct architecture behind the `AgentAdapter` interface."""

    def __init__(
        self,
        engine: Engine,
        llm: BaseChatModel,
        model_name: str,
        max_tool_steps: int = 6,
        recent_turns: int = 2,
    ) -> None:
        """Bind the tools once and remember how much prior-turn history to include."""
        self.id = f"monolithic-{model_name}"
        tools = build_tools(engine)
        self._llm_with_tools = llm.bind_tools(tools)
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._structured_llm = llm.with_structured_output(MonolithicResult)
        self._max_tool_steps = max_tool_steps
        self._recent_turns = recent_turns

    def _run_tool_loop(self, messages: list[BaseMessage]) -> list[LedgerEntry]:
        """Run the ReAct loop until the model stops requesting tools or the step
        limit is hit, appending every result to a ledger."""
        ledger: list[LedgerEntry] = []
        while True:
            response = self._llm_with_tools.invoke(messages)
            assert isinstance(response, AIMessage)
            messages.append(response)

            if not response.tool_calls or len(ledger) >= self._max_tool_steps:
                return ledger

            for tool_call in response.tool_calls:
                tool = self._tools_by_name[tool_call["name"]]
                try:
                    raw_result = tool.invoke(tool_call["args"])
                except CategoryNotFoundError as exc:
                    content = json.dumps(
                        {"error": str(exc), "valid_categories": exc.valid_categories}
                    )
                except ValidationError as exc:
                    content = json.dumps({"error": f"invalid arguments: {exc}"})
                else:
                    ledger_id = f"L{len(ledger) + 1}"
                    serialized = serialize_tool_result(raw_result)
                    ledger.append(
                        LedgerEntry(
                            ledger_id=ledger_id,
                            tool_name=tool_call["name"],
                            args=tool_call["args"],
                            result=serialized,
                            timestamp=datetime.now(UTC).isoformat(),
                        )
                    )
                    content = json.dumps({"ledger_id": ledger_id, "result": serialized})
                messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

    def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
        """Answer one question: run the tool loop, then make one final
        structured-output call that both routes and (if answering) drafts."""
        start = time.monotonic()
        recent = prior_turns[-self._recent_turns :] if self._recent_turns > 0 else []

        messages: list[BaseMessage] = [
            SystemMessage(content=MONOLITHIC_SYSTEM_PROMPT),
            *[
                message
                for trace in recent
                for message in (
                    HumanMessage(content=trace.question),
                    AIMessage(content=trace.final_answer or ""),
                )
            ],
            HumanMessage(content=question),
        ]
        ledger = self._run_tool_loop(messages)

        final_prompt = MONOLITHIC_FINAL_INSTRUCTIONS.format(
            history=_format_history(recent), question=question, ledger=_format_ledger(ledger)
        )
        result = self._structured_llm.invoke([HumanMessage(content=final_prompt)])
        assert isinstance(result, MonolithicResult)

        final_text = result.text
        groundedness_ok = True
        if result.route == "answer":
            check = verify(StructuredAnswer(text=result.text, claims=result.claims), ledger)
            groundedness_ok = check.ok
            if not groundedness_ok:
                final_text = GROUNDEDNESS_CAVEAT + result.text

        latency_ms = int((time.monotonic() - start) * 1000)
        return RunTrace(
            turn_id=str(len(prior_turns) + 1),
            question=question,
            resolved_question=question,  # no separate resolution step in this architecture
            route=result.route,
            ledger=ledger,
            draft_answer=StructuredAnswer(text=result.text, claims=result.claims),
            groundedness_ok=groundedness_ok,
            retries=0,  # this architecture never retries an ungrounded draft
            final_answer=final_text,
            latency_ms=latency_ms,
        )
