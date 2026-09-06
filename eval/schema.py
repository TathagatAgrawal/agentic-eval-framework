"""The test case schema and YAML loader for the eval harness's test set.

A `TestCase` is either single-turn (`question` + `expected`) or a multi-turn
sequence (`turns`, each with its own `expected`) -- never both. Every field here
is deliberately architecture-agnostic (see design/eval-harness-plan.md §2): a
test case only ever references `RunTrace` fields, never anything LangGraph- or
Gemini-specific, so the same test set can later drive a different `AgentAdapter`.
"""

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExpectedResult(BaseModel):
    """What a single turn's `RunTrace` is expected to look like."""

    behavior: Literal["answer", "clarify", "refuse"] = Field(
        description="The expected route: 'answer', 'clarify', or 'refuse'."
    )
    value: float | None = Field(
        default=None,
        description="A single expected numeric value, for behavior='answer' cases with one figure.",
    )
    values: list[float] | None = Field(
        default=None,
        description=(
            "Multiple expected numeric values, for behavior='answer' cases that state "
            "more than one figure (e.g. a comparison)."
        ),
    )
    tolerance: float = Field(default=0.01, description="Absolute tolerance for value/values.")
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tool names expected to appear somewhere in the turn's ledger.",
    )
    expected_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Partial match: every key/value here must appear in the flattened args of "
            "some ledger entry from this turn. Used to verify multi-turn reference "
            "resolution actually picked the right category/date_range."
        ),
    )
    optimal_tool_calls: int | None = Field(
        default=None,
        description=(
            "The fewest tool calls achievable with the current toolset for this turn "
            "(see design/eval-harness-plan.md §4.1); None for clarify/refuse turns, "
            "which aren't scored on efficiency."
        ),
    )
    groundedness_ok: bool = Field(
        default=True,
        description="Expected value of the agent's self-reported groundedness_ok.",
    )

    @model_validator(mode="after")
    def _check_answer_has_a_target_value(self) -> "ExpectedResult":
        """An 'answer' expectation must specify what number(s) to check for."""
        if self.behavior == "answer" and self.value is None and self.values is None:
            raise ValueError("behavior='answer' requires 'value' or 'values'")
        return self


class Turn(BaseModel):
    """One turn within a multi-turn test case."""

    question: str
    expected: ExpectedResult


class TestCase(BaseModel):
    """A single test case: either one question, or a multi-turn sequence."""

    __test__: ClassVar[bool] = False  # tell pytest this isn't a test class despite the name

    id: str
    category: str
    question: str | None = None
    expected: ExpectedResult | None = None
    turns: list[Turn] | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> "TestCase":
        """Enforce exactly one of (question + expected) or (turns) is set."""
        is_single_turn = self.question is not None
        is_multi_turn = self.turns is not None
        if is_single_turn == is_multi_turn:
            raise ValueError(
                f"test case '{self.id}' must set either (question + expected) or "
                "(turns), not both/neither"
            )
        if is_single_turn and self.expected is None:
            raise ValueError(f"test case '{self.id}' has a question but no expected result")
        if is_multi_turn and not self.turns:
            raise ValueError(f"test case '{self.id}' has an empty turns list")
        return self

    @property
    def type(self) -> Literal["single_turn", "multi_turn"]:
        """Whether this case is a single question or a multi-turn sequence."""
        return "multi_turn" if self.turns is not None else "single_turn"


def load_test_cases(*paths: Path) -> list[TestCase]:
    """Load and validate every test case from the given YAML files.

    Raises if two test cases (in the same file or across files) share an id.
    """
    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    for path in paths:
        raw_cases = yaml.safe_load(path.read_text()) or []
        for raw_case in raw_cases:
            case = TestCase.model_validate(raw_case)
            if case.id in seen_ids:
                raise ValueError(f"duplicate test case id '{case.id}' (found in {path})")
            seen_ids.add(case.id)
            cases.append(case)
    return cases
