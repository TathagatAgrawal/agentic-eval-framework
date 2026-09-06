"""Wraps this project's LangGraph agent as a concrete `AgentAdapter`.

This is the only place LangGraph specifics are allowed to leak past the
`AgentAdapter` interface (see `adapter.py`) -- the CLI and the eval harness
drive this through `run_turn` and never touch `build_graph`, `AgentState`, or
`TurnMemory` directly. `LangGraphAgentConfig` is likewise this architecture's
own configuration -- which model(s) it uses, its tool-loop limit, its
groundedness retry limit -- kept separate from the universal `Settings` in
`config.py`, per design/eval-harness-plan.md §2: a different architecture would
define its own config with its own fields instead of sharing this one.
"""

import time
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine

from finance_qna.agent.graph import build_graph
from finance_qna.agent.state import initial_state
from finance_qna.tracing.trace import RunTrace, build_trace, turn_memory_from_trace


class LangGraphAgentConfig(BaseSettings):
    """Configuration specific to the LangGraph architecture, loaded from
    `LANGGRAPH_*` environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LANGGRAPH_", extra="ignore")

    agent_model: str = "gemini-3.5-flash-lite"
    classifier_model: str = "gemini-3.5-flash-lite"
    max_tool_steps: int = 6
    groundedness_retry_limit: int = 1


@lru_cache
def get_langgraph_config() -> LangGraphAgentConfig:
    """Return the process-wide `LangGraphAgentConfig` instance, loaded once and cached."""
    return LangGraphAgentConfig()


class LangGraphAdapter:
    """Runs this project's LangGraph agent behind the `AgentAdapter` interface."""

    def __init__(
        self,
        engine: Engine,
        agent_llm: BaseChatModel,
        classifier_llm: BaseChatModel,
        model_name: str,
        max_tool_steps: int = 6,
        groundedness_retry_limit: int = 1,
        recent_turns: int = 2,
    ) -> None:
        """Compile the graph once, and remember how much prior-turn history to
        feed `contextualize`."""
        self.id = f"langgraph-{model_name}"
        self._graph: Any = build_graph(
            engine, agent_llm, classifier_llm, max_tool_steps, groundedness_retry_limit
        )
        self._recent_turns = recent_turns

    def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
        """Answer one question, converting recent `RunTrace`s into this graph's
        own `TurnMemory` for `contextualize`, then converting the result back."""
        recent = prior_turns[-self._recent_turns :] if self._recent_turns > 0 else []
        session_memory = [
            turn_memory_from_trace(trace, turn_id=i + 1)
            for i, trace in enumerate(recent)
            if trace.route == "answer"
        ]

        state = initial_state(question, session_memory=session_memory)
        start = time.monotonic()
        result = self._graph.invoke(state)
        latency_ms = int((time.monotonic() - start) * 1000)

        return build_trace(result, turn_id=str(len(prior_turns) + 1), latency_ms=latency_ms)
