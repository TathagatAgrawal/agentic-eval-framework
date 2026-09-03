"""Fake LLM test doubles so agent-graph tests never call the real Gemini API.

Each fake implements just enough of the `BaseChatModel` surface (`bind_tools`,
`with_structured_output`, `invoke`) for the node factories in `agent/nodes.py`
to work against it, returning pre-scripted responses in order.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel


class FakeToolCallingLLM:
    """A stand-in for the agent LLM: returns a fixed sequence of AIMessages."""

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        """Store the scripted responses to hand back on successive `invoke` calls."""
        self._responses = iter(responses)

    def bind_tools(self, tools: Any) -> "FakeToolCallingLLM":
        """Ignore the tool list (the fake doesn't call a real model) and return self."""
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return the next scripted AIMessage, regardless of the input messages."""
        return next(self._responses)


class FakeStructuredLLM:
    """A stand-in for the draft_answer LLM: returns a fixed structured-output value."""

    def __init__(self, answer: BaseModel) -> None:
        """Store the scripted answer to hand back on `invoke`."""
        self._answer = answer

    def with_structured_output(self, schema: Any) -> "FakeStructuredLLM":
        """Ignore the requested schema (the fake always returns its scripted answer)."""
        return self

    def invoke(self, messages: list[BaseMessage]) -> BaseModel:
        """Return the scripted structured answer, regardless of the input messages."""
        return self._answer


class FakeAgentLLM:
    """Combines `FakeToolCallingLLM` and `FakeStructuredLLM` so one fake can stand in
    for the single `llm` the full graph binds tools on and gets structured output from."""

    def __init__(self, tool_call_responses: Sequence[AIMessage], final_answer: BaseModel) -> None:
        """Store the scripted act-node responses and the scripted final structured answer."""
        self._tool_call_responses = iter(tool_call_responses)
        self._final_answer = final_answer

    def bind_tools(self, tools: Any) -> "_BoundFakeAgentLLM":
        """Return a view of this fake that serves scripted AIMessages on `invoke`."""
        return _BoundFakeAgentLLM(self._tool_call_responses)

    def with_structured_output(self, schema: Any) -> "_StructuredFakeAgentLLM":
        """Return a view of this fake that serves the scripted final answer on `invoke`."""
        return _StructuredFakeAgentLLM(self._final_answer)


class _BoundFakeAgentLLM:
    """The `bind_tools(...)` result for `FakeAgentLLM`."""

    def __init__(self, responses: Any) -> None:
        """Store the shared iterator of scripted AIMessages."""
        self._responses = responses

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return the next scripted AIMessage."""
        return next(self._responses)


class _StructuredFakeAgentLLM:
    """The `with_structured_output(...)` result for `FakeAgentLLM`."""

    def __init__(self, answer: BaseModel) -> None:
        """Store the scripted final answer."""
        self._answer = answer

    def invoke(self, messages: list[BaseMessage]) -> BaseModel:
        """Return the scripted final answer."""
        return self._answer
