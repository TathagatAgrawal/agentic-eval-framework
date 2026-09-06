"""Tests for the regression store, against hand-built RunRecords -- no adapter,
no agent, no LLM, per design/eval-harness-plan.md's build order step 6.
"""

from datetime import UTC, datetime
from pathlib import Path

from eval.results import RunRecord, TestCaseResult
from eval.store import load_all_runs, save_run


def _run_record(run_id: str, timestamp: datetime) -> RunRecord:
    """Build a minimal RunRecord fixture."""
    return RunRecord(
        run_id=run_id,
        timestamp=timestamp,
        agent_id="langgraph-fake-model",
        label="test",
        results=[
            TestCaseResult(
                case_id="simple_001",
                category="simple_lookup",
                scores={"numeric_correctness": True},
                optimal_tool_calls=1,
                actual_tool_calls=1,
                overage=0,
                retries=0,
                trace_refs=["runs/eval-1/simple_001/1.json"],
            )
        ],
        summary={"overall.numeric_correctness": 1.0},
    )


def test_save_run_writes_a_json_file(tmp_path: Path) -> None:
    """save_run must write a file named after the run's id under runs_dir."""
    record = _run_record("run-1", datetime(2025, 1, 1, tzinfo=UTC))

    path = save_run(record, runs_dir=tmp_path)

    assert path == tmp_path / "run-1.json"
    assert path.exists()


def test_load_all_runs_reads_back_saved_runs(tmp_path: Path) -> None:
    """Every run saved to runs_dir must come back from load_all_runs."""
    record_a = _run_record("run-a", datetime(2025, 1, 1, tzinfo=UTC))
    record_b = _run_record("run-b", datetime(2025, 1, 2, tzinfo=UTC))
    save_run(record_a, runs_dir=tmp_path)
    save_run(record_b, runs_dir=tmp_path)

    loaded = load_all_runs(runs_dir=tmp_path)

    assert {r.run_id for r in loaded} == {"run-a", "run-b"}


def test_load_all_runs_returns_them_oldest_first(tmp_path: Path) -> None:
    """Runs must come back sorted by timestamp, regardless of save order."""
    newer = _run_record("newer", datetime(2025, 6, 1, tzinfo=UTC))
    older = _run_record("older", datetime(2025, 1, 1, tzinfo=UTC))
    save_run(newer, runs_dir=tmp_path)
    save_run(older, runs_dir=tmp_path)

    loaded = load_all_runs(runs_dir=tmp_path)

    assert [r.run_id for r in loaded] == ["older", "newer"]


def test_load_all_runs_returns_empty_list_when_no_runs_saved_yet(tmp_path: Path) -> None:
    """load_all_runs must return an empty list, not raise, when runs_dir doesn't exist."""
    loaded = load_all_runs(runs_dir=tmp_path / "does-not-exist")
    assert loaded == []
