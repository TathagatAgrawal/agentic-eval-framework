"""Tests for LangGraphAgentConfig, LangGraphAdapter, and the build_adapter factory.

The LLM is mocked throughout (see tests/fakes.py) so this suite never calls the
real Gemini API.
"""

from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from finance_qna.agent.adapter import build_adapter
from finance_qna.agent.answer import StructuredAnswer
from finance_qna.agent.contextualize import ContextualizeResult
from finance_qna.agent.langgraph_adapter import LangGraphAdapter, LangGraphAgentConfig
from finance_qna.agent.route import RouteDecision
from finance_qna.config import Settings
from tests.fakes import FakeLLM


def test_langgraph_agent_config_defaults() -> None:
    """LangGraphAgentConfig must load with sensible defaults with no env overrides."""
    config = LangGraphAgentConfig(_env_file=None)
    assert config.agent_model == "gemini-3.5-flash-lite"
    assert config.classifier_model == "gemini-3.5-flash-lite"
    assert config.max_tool_steps == 6
    assert config.groundedness_retry_limit == 1


def test_build_adapter_rejects_unknown_architecture() -> None:
    """build_adapter must raise on an unrecognized agent_architecture value."""
    settings = Settings(google_api_key="fake-key", agent_architecture="some-other-architecture")  # type: ignore[call-arg]
    try:
        build_adapter(settings)
    except ValueError as exc:
        assert "some-other-architecture" in str(exc)
    else:
        raise AssertionError("expected build_adapter to raise ValueError")


def _fake_llm_for_one_turn(question: str, answer: StructuredAnswer) -> FakeLLM:
    """Build a FakeLLM scripted to resolve `question` unambiguously, route it to
    'answer', call no tools, and produce `answer` as the draft."""
    return FakeLLM(
        structured_responses={
            "ContextualizeResult": [
                ContextualizeResult(resolved_question=question, is_ambiguous=False)
            ],
            "RouteDecision": [RouteDecision(route="answer", reason="unambiguous")],
            "StructuredAnswer": [answer],
        },
        tool_call_responses=[AIMessage(content="I already have enough information.")],
    )


def test_langgraph_adapter_run_turn_produces_a_run_trace(fixture_engine: Engine) -> None:
    """run_turn must return a RunTrace with the adapter's id-prefixed model name."""
    question = "How much did I spend on dining?"
    answer = StructuredAnswer(text="You spent nothing new.", claims=[])
    fake_llm = _fake_llm_for_one_turn(question, answer)

    adapter = LangGraphAdapter(
        engine=fixture_engine,
        agent_llm=fake_llm,  # type: ignore[arg-type]
        classifier_llm=fake_llm,  # type: ignore[arg-type]
        model_name="fake-model",
    )

    trace = adapter.run_turn(question, prior_turns=[])

    assert adapter.id == "langgraph-fake-model"
    assert trace.turn_id == "1"
    assert trace.resolved_question == question
    assert trace.route == "answer"
    assert trace.final_answer == answer.text


def test_langgraph_adapter_only_carries_forward_answered_turns(fixture_engine: Engine) -> None:
    """A prior turn whose route wasn't 'answer' must not be converted into
    session_memory (nothing was actually resolved on that turn)."""
    from finance_qna.tracing.trace import RunTrace

    unanswered_prior_turn = RunTrace(
        turn_id="1",
        question="How much did I spend this month?",
        resolved_question="How much did I spend this month?",
        route="clarify",
        ledger=[],
        draft_answer=None,
        groundedness_ok=False,
        retries=0,
        final_answer="Which month did you mean?",
        latency_ms=10,
    )

    question = "How much did I spend on dining in July 2024?"
    # no tool call is made in this scripted turn, so claims must stay empty --
    # citing a ledger id with nothing in the ledger would fail groundedness and
    # trigger a retry this fake LLM isn't scripted to handle
    answer = StructuredAnswer(text="You spent $415.77 on dining.", claims=[])
    fake_llm = _fake_llm_for_one_turn(question, answer)

    adapter = LangGraphAdapter(
        engine=fixture_engine,
        agent_llm=fake_llm,  # type: ignore[arg-type]
        classifier_llm=fake_llm,  # type: ignore[arg-type]
        model_name="fake-model",
    )

    trace = adapter.run_turn(question, prior_turns=[unanswered_prior_turn])

    assert trace.turn_id == "2"
    assert trace.route == "answer"
