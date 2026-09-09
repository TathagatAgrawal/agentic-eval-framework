"""The Plan-and-Execute architecture: `contextualize` and `route` are reused
unchanged from the LangGraph baseline; only the interleaved act/tool_node loop
is replaced with an upfront batch plan, deterministic execution, and (if the
draft is ungrounded) a bounded re-plan -- never an open-ended reasoning loop.

See design/alternative-architectures-plan.md §3 for the full rationale: this
isolates "plan upfront vs. reason interleaved" as a single variable against the
baseline, since every other stage is held identical.

Implementation note: the "plan" step uses `llm.bind_tools(...)` (the same
native function-calling every other architecture uses for tool calls) rather
than a custom structured-output `Plan` schema with a generic `args: dict`
field. An earlier version tried the latter; live testing showed
gemini-3.5-flash-lite reliably returned an empty plan for it (confirmed twice,
with an explicit example in the prompt making no difference), and inspecting
the schema showed why: an `additionalProperties: true` open-ended object is a
much harder target for constrained JSON decoding to populate than a schema the
model was actually trained to fill in. Native function-calling doesn't have
that problem, and a single (non-looped) call to it can still return several
tool calls at once -- which is what "plan upfront, don't interleave" needs.
"""

import json
import time
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine

from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.contextualize import ContextualizeResult
from finance_qna.agent.groundedness import verify
from finance_qna.agent.nodes import extract_text, format_session_memory, serialize_tool_result
from finance_qna.agent.prompts import (
    CLARIFY_INSTRUCTIONS,
    CONTEXTUALIZE_INSTRUCTIONS,
    DRAFT_ANSWER_INSTRUCTIONS,
    GROUNDEDNESS_CAVEAT,
    PLAN_INSTRUCTIONS,
    REFUSE_INSTRUCTIONS,
    REPLAN_INSTRUCTIONS,
    ROUTE_INSTRUCTIONS,
)
from finance_qna.agent.route import RouteDecision
from finance_qna.agent.state import LedgerEntry
from finance_qna.tools.errors import CategoryNotFoundError
from finance_qna.tools.registry import build_tools
from finance_qna.tracing.trace import RunTrace, turn_memory_from_trace


class PlanExecuteConfig(BaseSettings):
    """Configuration specific to the Plan-and-Execute architecture, loaded from
    `PLAN_EXECUTE_*` environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PLAN_EXECUTE_", extra="ignore")

    model: str = "gemini-3.5-flash-lite"
    groundedness_retry_limit: int = 1


@lru_cache
def get_plan_execute_config() -> PlanExecuteConfig:
    """Return the process-wide `PlanExecuteConfig` instance, loaded once and cached."""
    return PlanExecuteConfig()


def _format_ledger(ledger: list[LedgerEntry]) -> str:
    """Render the ledger as text for the draft/replan prompts."""
    if not ledger:
        return "(no tool calls were made)"
    return "\n".join(
        f"{entry['ledger_id']} ({entry['tool_name']}): {json.dumps(entry['result'])}"
        for entry in ledger
    )


class PlanExecuteAdapter:
    """Runs the Plan-and-Execute architecture behind the `AgentAdapter` interface."""

    def __init__(
        self,
        engine: Engine,
        llm: BaseChatModel,
        model_name: str,
        groundedness_retry_limit: int = 1,
        recent_turns: int = 2,
    ) -> None:
        """Bind the tools (for both planning and execution) and every
        structured-output role."""
        self.id = f"plan_execute-{model_name}"
        self._llm = llm
        tools = build_tools(engine)
        self._llm_with_tools = llm.bind_tools(tools)
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._contextualize_llm = llm.with_structured_output(ContextualizeResult)
        self._route_llm = llm.with_structured_output(RouteDecision)
        self._draft_llm = llm.with_structured_output(StructuredAnswer)
        self._retry_limit = groundedness_retry_limit
        self._recent_turns = recent_turns

    def _plan_and_execute(self, prompt_text: str, ledger: list[LedgerEntry]) -> None:
        """Ask for every needed tool call in a single turn (native
        function-calling can return several at once), then execute all of them
        -- no loop back to the model to observe a result before deciding on
        the next call, which is what distinguishes this from interleaved ReAct."""
        response = self._llm_with_tools.invoke([HumanMessage(content=prompt_text)])
        assert isinstance(response, AIMessage)

        for tool_call in response.tool_calls:
            tool = self._tools_by_name.get(tool_call["name"])
            if tool is None:
                continue
            try:
                raw_result = tool.invoke(tool_call["args"])
            except (CategoryNotFoundError, ValidationError):
                continue
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

    def _draft(self, resolved_question: str, ledger: list[LedgerEntry]) -> StructuredAnswer:
        """Draft the answer from the current ledger, exactly like the baseline."""
        prompt = DRAFT_ANSWER_INSTRUCTIONS.format(
            question=resolved_question, ledger=_format_ledger(ledger)
        )
        result = self._draft_llm.invoke([HumanMessage(content=prompt)])
        assert isinstance(result, StructuredAnswer)
        return result

    def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
        """Resolve and route the question (as the baseline does), then plan,
        execute, draft, and -- if ungrounded -- re-plan up to the retry limit."""
        start = time.monotonic()
        recent = prior_turns[-self._recent_turns :] if self._recent_turns > 0 else []
        session_memory = [
            turn_memory_from_trace(trace, turn_id=i + 1)
            for i, trace in enumerate(recent)
            if trace.route == "answer"
        ]

        contextualize_prompt = CONTEXTUALIZE_INSTRUCTIONS.format(
            question=question, memory=format_session_memory(session_memory)
        )
        ctx_result = self._contextualize_llm.invoke([HumanMessage(content=contextualize_prompt)])
        assert isinstance(ctx_result, ContextualizeResult)
        resolved_question = ctx_result.resolved_question

        route: Literal["answer", "clarify", "refuse"]
        if ctx_result.is_ambiguous:
            route = "clarify"
            route_reason = ctx_result.ambiguity_reason or "the question is ambiguous"
        else:
            route_prompt = ROUTE_INSTRUCTIONS.format(question=resolved_question)
            route_decision = self._route_llm.invoke([HumanMessage(content=route_prompt)])
            assert isinstance(route_decision, RouteDecision)
            route = route_decision.route
            route_reason = route_decision.reason

        ledger: list[LedgerEntry] = []
        retries = 0
        groundedness_ok = True

        if route == "clarify":
            prompt = CLARIFY_INSTRUCTIONS.format(question=question, reason=route_reason)
            final_text = extract_text(self._llm.invoke([HumanMessage(content=prompt)]).content)
            draft = StructuredAnswer(text=final_text, claims=[])
        elif route == "refuse":
            prompt = REFUSE_INSTRUCTIONS.format(question=question, reason=route_reason)
            final_text = extract_text(self._llm.invoke([HumanMessage(content=prompt)]).content)
            draft = StructuredAnswer(text=final_text, claims=[])
        else:
            plan_prompt = PLAN_INSTRUCTIONS.format(question=resolved_question)
            self._plan_and_execute(plan_prompt, ledger)

            draft = self._draft(resolved_question, ledger)
            check = verify(draft, ledger)
            groundedness_ok = check.ok

            while not groundedness_ok and retries < self._retry_limit:
                retries += 1
                unsupported_desc = "; ".join(
                    f"{claim.value} (cited {claim.ledger_id})" for claim in check.unsupported
                )
                replan_prompt = REPLAN_INSTRUCTIONS.format(
                    ledger=_format_ledger(ledger),
                    unsupported=unsupported_desc,
                    question=resolved_question,
                )
                self._plan_and_execute(replan_prompt, ledger)

                draft = self._draft(resolved_question, ledger)
                check = verify(draft, ledger)
                groundedness_ok = check.ok

            final_text = draft.text if groundedness_ok else GROUNDEDNESS_CAVEAT + draft.text

        latency_ms = int((time.monotonic() - start) * 1000)
        return RunTrace(
            turn_id=str(len(prior_turns) + 1),
            question=question,
            resolved_question=resolved_question,
            route=route,
            ledger=ledger,
            draft_answer=draft,
            groundedness_ok=groundedness_ok,
            retries=retries,
            final_answer=final_text,
            latency_ms=latency_ms,
        )
