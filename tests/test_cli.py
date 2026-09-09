"""Tests for the CLI's `data generate` and `eval run` commands.

`ask` and `chat` are not covered here since they require a real Gemini call to
exercise meaningfully -- they're verified manually/live instead, per the
project's directive to keep the automated test suite free of API calls.
`eval run` is covered with a FakeAdapter (no LLM) against a temp copy of the
real testset, isolated from the repo's actual `eval/runs/`.
"""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from finance_qna.cli.main import app
from finance_qna.tracing.trace import RunTrace
from tests.fakes import FakeAdapter

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_data_generate_creates_database_and_ground_truth(tmp_path: Path) -> None:
    """The `data generate` command must write both the DB and its ground-truth sidecar."""
    db_path = tmp_path / "test.db"

    result = runner.invoke(app, ["data", "generate", "--seed", "7", "--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert (db_path.parent / "ground_truth.json").exists()
    assert "seed=7" in result.output


def test_eval_run_writes_a_run_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`finance-qna eval run` must run every case through the adapter and save a RunRecord.

    Isolated from the real repo: `_repo_root` is monkeypatched to a tmp dir
    (with a copy of the real single_turn testset), and `build_adapter` is
    monkeypatched to a FakeAdapter, so this never touches the real
    `eval/runs/`, the real `runs/`, or the Gemini API.
    """
    testset_dir = tmp_path / "eval" / "testset"
    testset_dir.mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "eval" / "testset" / "single_turn.yaml", testset_dir / "single_turn.yaml"
    )

    generic_trace = RunTrace(
        turn_id="1",
        question="q",
        resolved_question="q",
        route="answer",
        ledger=[],
        draft_answer=None,
        groundedness_ok=True,
        retries=0,
        final_answer="a",
        latency_ms=1,
    )
    fake_adapter = FakeAdapter(id="fake-cli", responses=[generic_trace] * 12)

    monkeypatch.setattr("finance_qna.cli.main._repo_root", lambda: tmp_path)
    monkeypatch.setattr("finance_qna.cli.main.build_adapter", lambda settings: fake_adapter)

    result = runner.invoke(app, ["eval", "run", "--suite", "single_turn"])

    assert result.exit_code == 0, result.output
    assert "agent=fake-cli" in result.output
    saved_runs = list((tmp_path / "eval" / "runs").glob("*.json"))
    assert len(saved_runs) == 1


def test_eval_run_limit_caps_the_number_of_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--limit N` must run only the first N cases, e.g. to stay under a rate limit."""
    testset_dir = tmp_path / "eval" / "testset"
    testset_dir.mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "eval" / "testset" / "single_turn.yaml", testset_dir / "single_turn.yaml"
    )

    generic_trace = RunTrace(
        turn_id="1",
        question="q",
        resolved_question="q",
        route="answer",
        ledger=[],
        draft_answer=None,
        groundedness_ok=True,
        retries=0,
        final_answer="a",
        latency_ms=1,
    )
    fake_adapter = FakeAdapter(id="fake-cli", responses=[generic_trace] * 5)

    monkeypatch.setattr("finance_qna.cli.main._repo_root", lambda: tmp_path)
    monkeypatch.setattr("finance_qna.cli.main.build_adapter", lambda settings: fake_adapter)

    result = runner.invoke(app, ["eval", "run", "--suite", "single_turn", "--limit", "5"])

    assert result.exit_code == 0, result.output
    assert "(5 cases" in result.output
    assert len(fake_adapter.calls) == 5


def test_eval_run_delay_defaults_to_env_var_and_can_be_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`eval run` must default its inter-case delay to EVAL_CASE_DELAY_SECONDS,
    and let `--delay` override that default."""
    testset_dir = tmp_path / "eval" / "testset"
    testset_dir.mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "eval" / "testset" / "single_turn.yaml", testset_dir / "single_turn.yaml"
    )

    generic_trace = RunTrace(
        turn_id="1",
        question="q",
        resolved_question="q",
        route="answer",
        ledger=[],
        draft_answer=None,
        groundedness_ok=True,
        retries=0,
        final_answer="a",
        latency_ms=1,
    )
    fake_adapter = FakeAdapter(id="fake-cli", responses=[generic_trace] * 5)
    seen_delays: list[float] = []

    def _spy_run_eval(adapter, cases, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        """Record the delay_seconds the CLI passed, without running a real eval."""
        seen_delays.append(kwargs["delay_seconds"])
        from eval.results import RunRecord

        return RunRecord(
            run_id="spy",
            timestamp="2025-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            agent_id=adapter.id,
            label=kwargs.get("label", ""),
            results=[],
            summary={},
        )

    monkeypatch.setattr("finance_qna.cli.main._repo_root", lambda: tmp_path)
    monkeypatch.setattr("finance_qna.cli.main.build_adapter", lambda settings: fake_adapter)
    monkeypatch.setattr("eval.runner.run_eval", _spy_run_eval)
    monkeypatch.setenv("EVAL_CASE_DELAY_SECONDS", "3.5")

    runner.invoke(app, ["eval", "run", "--suite", "single_turn", "--limit", "1"])
    runner.invoke(app, ["eval", "run", "--suite", "single_turn", "--limit", "1", "--delay", "0"])

    assert seen_delays == [3.5, 0.0]
