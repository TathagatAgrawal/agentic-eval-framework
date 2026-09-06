"""Fake test doubles so tests never call the real Gemini API.

`FakeLLM` implements just enough of the `BaseChatModel` surface (`bind_tools`,
`with_structured_output`, `invoke`) for the node factories in `agent/nodes.py`
to work against it, returning pre-scripted responses in order. A single class
covers every node because the same `llm` instance is bound differently by
different nodes (tool-calling for `act`, structured output for `route` and
`draft_answer`, plain `invoke` for `clarify`/`refuse`).

`FakeAdapter` implements the `AgentAdapter` protocol directly, one level above
`FakeLLM` -- it lets eval-runner tests exercise `run_test_case`/`run_eval`
without any agent, graph, or LLM at all, per design/eval-harness-plan.md's
build-order note that the runner should be testable against a fake adapter.
"""

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from finance_qna.tracing.trace import RunTrace


class FakeLLM:
    """Serves scripted responses to each of the three ways a node can call an LLM."""

    def __init__(
        self,
        *,
        tool_call_responses: Sequence[AIMessage] = (),
        invoke_responses: Sequence[AIMessage] = (),
        structured_responses: dict[str, Sequence[BaseModel]] | None = None,
    ) -> None:
        """Store scripted responses per call style, keyed by schema name for
        structured-output calls (since one fake serves every node's schema)."""
        self._tool_call_responses: Iterator[AIMessage] = iter(tool_call_responses)
        self._invoke_responses: Iterator[AIMessage] = iter(invoke_responses)
        self._structured_responses: dict[str, Iterator[BaseModel]] = {
            name: iter(responses) for name, responses in (structured_responses or {}).items()
        }

    def bind_tools(self, tools: Any) -> "_BoundToolLLM":
        """Return a view that serves scripted AIMessages for the act node's tool loop."""
        return _BoundToolLLM(self._tool_call_responses)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return the next scripted plain-text AIMessage (used by clarify/refuse)."""
        return next(self._invoke_responses)

    def with_structured_output(self, schema: type[BaseModel]) -> "_StructuredLLM":
        """Return a view that serves the scripted response for this schema type.

        Binding happens at graph-construction time for every node regardless of
        which branch a test actually exercises, so a schema with no scripted
        responses is left as an empty iterator rather than raising here -- it
        only raises (via StopIteration) if a test's graph run actually reaches
        a node that tries to invoke it.
        """
        return _StructuredLLM(self._structured_responses.get(schema.__name__, iter(())))


class _BoundToolLLM:
    """The `bind_tools(...)` result: serves the act node's scripted AIMessages."""

    def __init__(self, responses: Iterator[AIMessage]) -> None:
        """Store the shared iterator of scripted AIMessages."""
        self._responses = responses

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return the next scripted AIMessage."""
        return next(self._responses)


class _StructuredLLM:
    """The `with_structured_output(...)` result for one schema type."""

    def __init__(self, responses: Iterator[BaseModel]) -> None:
        """Store the shared iterator of scripted structured responses."""
        self._responses = responses

    def invoke(self, messages: list[BaseMessage]) -> BaseModel:
        """Return the next scripted structured response."""
        return next(self._responses)


class FakeAdapter:
    """A scripted `AgentAdapter`: returns pre-built `RunTrace`s in order,
    regardless of the question or `prior_turns` passed in."""

    def __init__(self, id: str, responses: Sequence[RunTrace]) -> None:
        """Store the adapter id and the scripted RunTraces to return in order."""
        self.id = id
        self._responses: Iterator[RunTrace] = iter(responses)
        self.calls: list[tuple[str, list[RunTrace]]] = []

    def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
        """Record the call and return the next scripted RunTrace."""
        self.calls.append((question, list(prior_turns)))
        return next(self._responses)
