"""The Context-Stuffing architecture: no tools at all. A precomputed summary of
the transaction data (monthly totals per category, labeled subscription price
changes) is injected directly into the draft-answer prompt every turn, wrapped
as one synthetic ledger entry so the same `groundedness.verify()` used by every
other architecture can check its claims without special-casing it.

`contextualize` and `route` are reused unchanged from the LangGraph baseline's
prompts/schemas -- only the tool-calling stage is replaced. See
design/alternative-architectures-plan.md §4 for the full rationale, including
the known, deliberate limitations (weaker groundedness checking, structurally
not-applicable contextual correctness) called out in §8.
"""

import json
import time
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine

from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.contextualize import ContextualizeResult
from finance_qna.agent.groundedness import verify
from finance_qna.agent.nodes import extract_text, format_session_memory
from finance_qna.agent.prompts import (
    CLARIFY_INSTRUCTIONS,
    CONTEXTUALIZE_INSTRUCTIONS,
    DRAFT_ANSWER_INSTRUCTIONS,
    GROUNDEDNESS_CAVEAT,
    REFUSE_INSTRUCTIONS,
    ROUTE_INSTRUCTIONS,
)
from finance_qna.agent.route import RouteDecision
from finance_qna.agent.state import LedgerEntry
from finance_qna.tools.models import DateRange, TransactionFilter
from finance_qna.tools.query_tools import aggregate_spending, list_categories
from finance_qna.tools.trend_tools import detect_subscription_changes
from finance_qna.tracing.trace import RunTrace, turn_memory_from_trace

# Wide enough to cover any realistic dataset without needing a MIN/MAX(date)
# query; only used to build the one-time precomputed context table.
_FULL_RANGE = DateRange(start=date(2000, 1, 1), end=date(2100, 1, 1))


def compute_context_table(engine: Engine) -> dict[str, Any]:
    """Precompute a monthly-totals-per-category summary plus labeled subscription
    price changes, once per adapter instance -- standing in for what a tool
    call would otherwise fetch on demand in the other architectures."""
    monthly_totals: dict[str, dict[str, str]] = {}
    for category in list_categories(engine):
        filt = TransactionFilter(category=category, date_range=_FULL_RANGE)
        results = aggregate_spending(engine, filt, group_by="month")
        assert isinstance(results, list)
        monthly_totals[category] = {
            result.group_key: str(result.total) for result in results if result.group_key
        }

    subscription_changes = [
        {
            "merchant": event.merchant,
            "old_amount": str(event.old_amount),
            "new_amount": str(event.new_amount),
            "change_date": event.change_date.isoformat(),
        }
        for event in detect_subscription_changes(engine, _FULL_RANGE)
    ]

    return {
        "monthly_category_totals": monthly_totals,
        "subscription_changes": subscription_changes,
    }


class ContextStuffingConfig(BaseSettings):
    """Configuration specific to the Context-Stuffing architecture, loaded from
    `CONTEXT_STUFFING_*` environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CONTEXT_STUFFING_", extra="ignore"
    )

    model: str = "gemini-3.5-flash-lite"


@lru_cache
def get_context_stuffing_config() -> ContextStuffingConfig:
    """Return the process-wide `ContextStuffingConfig` instance, loaded once and cached."""
    return ContextStuffingConfig()


class ContextStuffingAdapter:
    """Runs the Context-Stuffing architecture behind the `AgentAdapter` interface."""

    def __init__(
        self,
        engine: Engine,
        llm: BaseChatModel,
        model_name: str,
        recent_turns: int = 2,
    ) -> None:
        """Precompute the context table once and bind the structured-output roles."""
        self.id = f"context_stuffing-{model_name}"
        self._llm = llm
        self._contextualize_llm = llm.with_structured_output(ContextualizeResult)
        self._route_llm = llm.with_structured_output(RouteDecision)
        self._draft_llm = llm.with_structured_output(StructuredAnswer)
        self._context_table = compute_context_table(engine)
        self._recent_turns = recent_turns

    def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
        """Resolve the question, route it, then either clarify/refuse or draft an
        answer from the precomputed context table -- no tool calls, no retry."""
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
        groundedness_ok = True

        if route == "clarify":
            prompt = CLARIFY_INSTRUCTIONS.format(question=question, reason=route_reason)
            response = self._llm.invoke([HumanMessage(content=prompt)])
            final_text = extract_text(response.content)
            draft = StructuredAnswer(text=final_text, claims=[])
        elif route == "refuse":
            prompt = REFUSE_INSTRUCTIONS.format(question=question, reason=route_reason)
            response = self._llm.invoke([HumanMessage(content=prompt)])
            final_text = extract_text(response.content)
            draft = StructuredAnswer(text=final_text, claims=[])
        else:
            ledger = [
                LedgerEntry(
                    ledger_id="L1",
                    tool_name="context_table",
                    args={},
                    result=self._context_table,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            ]
            ledger_text = f"L1 (context_table): {json.dumps(self._context_table)}"
            draft_prompt = DRAFT_ANSWER_INSTRUCTIONS.format(
                question=resolved_question, ledger=ledger_text
            )
            draft_response = self._draft_llm.invoke([HumanMessage(content=draft_prompt)])
            assert isinstance(draft_response, StructuredAnswer)
            draft = draft_response

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
            retries=0,  # this architecture never retries an ungrounded draft
            final_answer=final_text,
            latency_ms=latency_ms,
        )
