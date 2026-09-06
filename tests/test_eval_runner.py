"""Tests for the eval runner, driven entirely by FakeAdapter -- no agent, no
graph, no LLM, per design/eval-harness-plan.md's build-order note that the
runner should be testable against a fake AgentAdapter.
"""

import json
from decimal import Decimal
from pathlib import Path

from eval.runner import run_eval, run_test_case, score_turn
from eval.schema import ExpectedResult, TestCase, Turn
from finance_qna.agent.answer import Claim, StructuredAnswer
from finance_qna.tracing.trace import RunTrace
from tests.fakes import FakeAdapter


def _trace(
    turn_id: str = "1",
    route: str | None = "answer",
    ledger: list | None = None,
    claims: list[Claim] | None = None,
    groundedness_ok: bool = True,
    retries: int = 0,
    final_answer: str | None = "an answer",
) -> RunTrace:
    """Build a RunTrace fixture with sensible defaults, overridable per test."""
    return RunTrace(
        turn_id=turn_id,
        question="a question",
        resolved_question="a question",
        route=route,
        ledger=ledger or [],
        draft_answer=StructuredAnswer(text=final_answer or "", claims=claims or []),
        groundedness_ok=groundedness_ok,
        retries=retries,
        final_answer=final_answer,
        latency_ms=100,
    )


def _ledger_entry(ledger_id: str, total: str = "1.0") -> dict:
    """Build a minimal ledger entry dict matching the LedgerEntry TypedDict shape,
    with `total` in its result so a claim citing this ledger_id can be grounded."""
    return {
        "ledger_id": ledger_id,
        "tool_name": "aggregate_spending_tool",
        "args": {},
        "result": {"total": total},
        "timestamp": "2025-01-01T00:00:00+00:00",
    }


def test_score_turn_runs_every_scorer() -> None:
    """score_turn must return a result for every metric plus the efficiency detail."""
    trace = _trace(
        claims=[Claim(value=Decimal("511.94"), ledger_id="L1")],
        ledger=[_ledger_entry("L1", total="511.94")],
    )
    expected = ExpectedResult(behavior="answer", value=511.94, tolerance=0.01, optimal_tool_calls=1)

    metrics, eff = score_turn(trace, expected)

    assert metrics["numeric_correctness"].passed is True
    assert metrics["groundedness_ok"].passed is True
    assert metrics["harness_groundedness_ok"].passed is True
    assert metrics["clarification"].passed is True  # correctly did NOT clarify
    assert metrics["refusal"].passed is True  # correctly did NOT refuse
    assert eff.efficient.passed is True


def test_run_test_case_single_turn_writes_one_trace(tmp_path: Path) -> None:
    """A single-turn case must call the adapter once and write exactly one trace file."""
    trace = _trace(
        claims=[Claim(value=Decimal("511.94"), ledger_id="L1")],
        ledger=[_ledger_entry("L1", total="511.94")],
    )
    adapter = FakeAdapter(id="fake-1", responses=[trace])
    case = TestCase(
        id="simple_001",
        category="simple_lookup",
        question="How much did I spend on groceries?",
        expected=ExpectedResult(
            behavior="answer", value=511.94, tolerance=0.01, optimal_tool_calls=1
        ),
    )

    result = run_test_case(adapter, case, tmp_path)

    assert result.case_id == "simple_001"
    assert result.scores["numeric_correctness"] is True
    assert result.actual_tool_calls == 1
    assert result.optimal_tool_calls == 1
    assert result.overage == 0
    assert len(result.trace_refs) == 1
    assert Path(result.trace_refs[0]).exists()
    assert len(adapter.calls) == 1
    assert adapter.calls[0] == ("How much did I spend on groceries?", [])


def test_run_test_case_multi_turn_threads_prior_turns(tmp_path: Path) -> None:
    """Each turn after the first must receive every prior turn's trace as prior_turns."""
    trace_1 = _trace(turn_id="1", claims=[], ledger=[_ledger_entry("L1")])
    trace_2 = _trace(turn_id="2", claims=[], ledger=[_ledger_entry("L1")])
    adapter = FakeAdapter(id="fake-1", responses=[trace_1, trace_2])
    case = TestCase(
        id="multi_001",
        category="multi_turn",
        turns=[
            Turn(question="Q1", expected=ExpectedResult(behavior="answer", value=1.0)),
            Turn(question="Q2", expected=ExpectedResult(behavior="answer", value=1.0)),
        ],
    )

    result = run_test_case(adapter, case, tmp_path)

    assert len(adapter.calls) == 2
    assert adapter.calls[0] == ("Q1", [])
    assert adapter.calls[1] == ("Q2", [trace_1])
    assert len(result.trace_refs) == 2
    assert result.actual_tool_calls == 2  # summed across both turns


def test_run_test_case_aggregates_scores_with_and_across_turns(tmp_path: Path) -> None:
    """The case-level score for a metric must fail if any turn where it applied failed."""
    passing_trace = _trace(
        turn_id="1",
        claims=[Claim(value=Decimal("1.0"), ledger_id="L1")],
        ledger=[_ledger_entry("L1")],
    )
    failing_trace = _trace(
        turn_id="2",
        claims=[Claim(value=Decimal("999.0"), ledger_id="L1")],
        ledger=[_ledger_entry("L1")],
    )
    adapter = FakeAdapter(id="fake-1", responses=[passing_trace, failing_trace])
    case = TestCase(
        id="multi_002",
        category="multi_turn",
        turns=[
            Turn(
                question="Q1", expected=ExpectedResult(behavior="answer", value=1.0, tolerance=0.01)
            ),
            Turn(
                question="Q2", expected=ExpectedResult(behavior="answer", value=1.0, tolerance=0.01)
            ),
        ],
    )

    result = run_test_case(adapter, case, tmp_path)

    assert result.scores["numeric_correctness"] is False


def test_run_test_case_clarify_case_leaves_numeric_and_efficiency_not_applicable(
    tmp_path: Path,
) -> None:
    """A clarify-expected case must not be scored on numeric correctness or efficiency."""
    trace = _trace(route="clarify", ledger=[], final_answer="Which month did you mean?")
    adapter = FakeAdapter(id="fake-1", responses=[trace])
    case = TestCase(
        id="ambiguous_001",
        category="ambiguous",
        question="How much did I spend this month?",
        expected=ExpectedResult(behavior="clarify"),
    )

    result = run_test_case(adapter, case, tmp_path)

    assert result.scores["clarification"] is True
    assert result.scores["numeric_correctness"] is None
    assert result.optimal_tool_calls is None
    assert result.overage is None


def test_run_eval_aggregates_a_run_record_with_summary(tmp_path: Path) -> None:
    """run_eval must produce a RunRecord whose summary reflects the mix of passes/fails."""
    passing_trace = _trace(
        claims=[Claim(value=Decimal("1.0"), ledger_id="L1")], ledger=[_ledger_entry("L1")]
    )
    failing_trace = _trace(
        claims=[Claim(value=Decimal("999.0"), ledger_id="L1")], ledger=[_ledger_entry("L1")]
    )
    adapter = FakeAdapter(id="fake-1", responses=[passing_trace, failing_trace])
    cases = [
        TestCase(
            id="case_pass",
            category="simple_lookup",
            question="Q1",
            expected=ExpectedResult(behavior="answer", value=1.0, tolerance=0.01),
        ),
        TestCase(
            id="case_fail",
            category="simple_lookup",
            question="Q2",
            expected=ExpectedResult(behavior="answer", value=1.0, tolerance=0.01),
        ),
    ]

    record = run_eval(adapter, cases, tmp_path, label="test-run")

    assert record.agent_id == "fake-1"
    assert record.label == "test-run"
    assert len(record.results) == 2
    assert record.summary["overall.numeric_correctness"] == 0.5
    assert record.summary["simple_lookup.numeric_correctness"] == 0.5


def test_run_test_case_records_an_error_instead_of_raising(tmp_path: Path) -> None:
    """A turn that raises (e.g. a 429) must produce an errored TestCaseResult,
    not propagate the exception."""
    adapter = FakeAdapter(id="fake-1", responses=[RuntimeError("429 rate limited")])
    case = TestCase(
        id="simple_001",
        category="simple_lookup",
        question="Q1",
        expected=ExpectedResult(behavior="answer", value=1.0, optimal_tool_calls=1),
    )

    result = run_test_case(adapter, case, tmp_path)

    assert result.error == "RuntimeError: 429 rate limited"
    assert result.scores == {}
    assert result.optimal_tool_calls is None
    assert result.trace_refs == []


def test_run_test_case_multi_turn_keeps_completed_turns_after_a_later_failure(
    tmp_path: Path,
) -> None:
    """If turn 2 of a sequence raises, turn 1's completed work must still be scored
    and returned, not discarded."""
    trace_1 = _trace(
        turn_id="1",
        claims=[Claim(value=Decimal("1.0"), ledger_id="L1")],
        ledger=[_ledger_entry("L1")],
    )
    adapter = FakeAdapter(id="fake-1", responses=[trace_1, RuntimeError("429 rate limited")])
    case = TestCase(
        id="multi_001",
        category="multi_turn",
        turns=[
            Turn(
                question="Q1", expected=ExpectedResult(behavior="answer", value=1.0, tolerance=0.01)
            ),
            Turn(question="Q2", expected=ExpectedResult(behavior="answer", value=1.0)),
        ],
    )

    result = run_test_case(adapter, case, tmp_path)

    assert result.error == "RuntimeError: 429 rate limited"
    assert len(result.trace_refs) == 1  # turn 1's trace was kept
    assert result.scores["numeric_correctness"] is True  # scored from turn 1 alone
    assert result.actual_tool_calls == 1


def test_run_eval_continues_past_a_failed_case_and_checkpoints_progress(tmp_path: Path) -> None:
    """One case failing must not stop the run, and every case (including the
    failure) must already be saved to disk by the time run_eval returns."""
    passing_trace = _trace(claims=[], ledger=[])
    adapter = FakeAdapter(
        id="fake-1",
        responses=[passing_trace, RuntimeError("429 rate limited"), passing_trace],
    )
    cases = [
        TestCase(
            id="case_1",
            category="c",
            question="Q1",
            expected=ExpectedResult(behavior="answer", value=1.0),
        ),
        TestCase(
            id="case_2",
            category="c",
            question="Q2",
            expected=ExpectedResult(behavior="answer", value=1.0),
        ),
        TestCase(
            id="case_3",
            category="c",
            question="Q3",
            expected=ExpectedResult(behavior="answer", value=1.0),
        ),
    ]
    runs_dir = tmp_path / "eval_runs"

    record = run_eval(adapter, cases, tmp_path / "traces", run_id="run-xyz", runs_dir=runs_dir)

    assert [r.case_id for r in record.results] == ["case_1", "case_2", "case_3"]
    assert record.results[1].error == "RuntimeError: 429 rate limited"
    assert len(adapter.calls) == 3  # case_3 still ran despite case_2 failing

    saved = json.loads((runs_dir / "run-xyz.json").read_text())
    assert len(saved["results"]) == 3
    assert saved["results"][1]["error"] == "RuntimeError: 429 rate limited"


def test_run_eval_saves_a_checkpoint_after_every_case(tmp_path: Path) -> None:
    """The saved RunRecord must reflect progress after each case, not only at the end."""

    class _RecordingAdapter(FakeAdapter):
        """A FakeAdapter that snapshots the on-disk run record after each call."""

        def __init__(self, id: str, responses: list, runs_dir: Path) -> None:
            """Store the runs_dir to inspect and the usual scripted responses."""
            super().__init__(id=id, responses=responses)
            self._runs_dir = runs_dir
            self.saved_case_counts: list[int] = []

        def run_turn(self, question: str, prior_turns: list[RunTrace]) -> RunTrace:
            """Record how many cases were saved to disk *before* this call runs."""
            saved_files = list(self._runs_dir.glob("*.json"))
            if saved_files:
                saved = json.loads(saved_files[0].read_text())
                self.saved_case_counts.append(len(saved["results"]))
            else:
                self.saved_case_counts.append(0)
            return super().run_turn(question, prior_turns)

    runs_dir = tmp_path / "eval_runs"
    trace = _trace(claims=[], ledger=[])
    adapter = _RecordingAdapter(id="fake-1", responses=[trace, trace, trace], runs_dir=runs_dir)
    cases = [
        TestCase(
            id=f"case_{i}",
            category="c",
            question=f"Q{i}",
            expected=ExpectedResult(behavior="answer", value=1.0),
        )
        for i in range(3)
    ]

    run_eval(adapter, cases, tmp_path / "traces", run_id="run-xyz", runs_dir=runs_dir)

    # before case 1 runs: nothing saved yet; before case 2: case 1's result is
    # already on disk; before case 3: cases 1-2 are on disk
    assert adapter.saved_case_counts == [0, 1, 2]
