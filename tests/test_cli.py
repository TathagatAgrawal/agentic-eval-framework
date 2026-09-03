"""Tests for the CLI's `data generate` command.

`ask` and `chat` are not covered here since they require a real Gemini call to
exercise meaningfully -- they're verified manually/live instead, per the
project's directive to keep the automated test suite free of API calls.
"""

from pathlib import Path

from typer.testing import CliRunner

from finance_qna.cli.main import app

runner = CliRunner()


def test_data_generate_creates_database_and_ground_truth(tmp_path: Path) -> None:
    """The `data generate` command must write both the DB and its ground-truth sidecar."""
    db_path = tmp_path / "test.db"

    result = runner.invoke(app, ["data", "generate", "--seed", "7", "--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert (db_path.parent / "ground_truth.json").exists()
    assert "seed=7" in result.output
