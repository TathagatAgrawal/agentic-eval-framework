"""The `AgentAdapter` protocol: the model- and architecture-agnostic seam both
the CLI and the eval harness drive an agent through.

Per design/eval-harness-plan.md §2, neither the CLI nor the eval harness should
need to know LangGraph or Gemini exist -- they only ever call `run_turn` and
exchange `RunTrace`s. `build_adapter` is the single place that picks a concrete
implementation based on configuration, so adding a new architecture means
adding a branch here, not touching the CLI or the eval runner. See
design/alternative-architectures-plan.md for the architectures beyond the
original `langgraph` baseline.
"""

from typing import Protocol

from finance_qna.agent.langgraph_adapter import LangGraphAdapter, get_langgraph_config
from finance_qna.agent.monolithic_adapter import MonolithicReactAdapter, get_monolithic_config
from finance_qna.config import Settings, get_llm
from finance_qna.data.db import get_engine
from finance_qna.tracing.trace import RunTrace


class AgentAdapter(Protocol):
    """Anything the CLI or eval harness can drive and score, regardless of
    implementation."""

    id: str

    def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
        """Answer one question, given the `RunTrace`s of prior turns in this session."""
        ...


def build_adapter(settings: Settings) -> AgentAdapter:
    """Construct the `AgentAdapter` selected by `settings.agent_architecture`."""
    engine = get_engine(settings.db_path)

    if settings.agent_architecture == "langgraph":
        config = get_langgraph_config()
        return LangGraphAdapter(
            engine=engine,
            agent_llm=get_llm(config.agent_model),
            classifier_llm=get_llm(config.classifier_model),
            model_name=config.agent_model,
            max_tool_steps=config.max_tool_steps,
            groundedness_retry_limit=config.groundedness_retry_limit,
        )

    if settings.agent_architecture == "monolithic":
        mono_config = get_monolithic_config()
        return MonolithicReactAdapter(
            engine=engine,
            llm=get_llm(mono_config.model),
            model_name=mono_config.model,
            max_tool_steps=mono_config.max_tool_steps,
        )

    raise ValueError(f"unknown agent architecture: {settings.agent_architecture!r}")
