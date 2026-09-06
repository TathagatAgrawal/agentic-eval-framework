"""The command-line interface: `data generate` builds the synthetic dataset,
`ask` answers one question, `chat` starts an interactive multi-turn session,
and `eval run`/`eval dashboard` drive the evaluation harness.

`ask`, `chat`, and `eval run` all drive the agent exclusively through
`AgentAdapter` (see `agent/adapter.py`) -- the CLI never touches LangGraph,
`AgentState`, or `TurnMemory` directly, and which model/architecture runs is
entirely a function of `Settings` (`agent_architecture` in `.env`, plus that
architecture's own config), not CLI code. Per the project's non-goals, this is
the whole interface -- no web UI, and `chat` sessions are never persisted
across process runs.
"""

import sys
import uuid
from pathlib import Path

import typer

from finance_qna.agent.adapter import build_adapter
from finance_qna.config import get_settings
from finance_qna.data.generate import generate as generate_dataset
from finance_qna.tracing.trace import RunTrace, write_trace

app = typer.Typer(help="Personal financial statement Q&A agent.")
data_app = typer.Typer(help="Manage the synthetic transaction dataset.")
eval_app = typer.Typer(help="Run and inspect the evaluation harness.")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")


def _repo_root() -> Path:
    """Return the repository root (cli -> finance_qna -> src -> root, four levels up).

    `eval/` lives here too, alongside `src/`. Only the `eval` subcommands need
    this -- `ask`/`chat`/`data generate` don't reach outside the installed package.
    """
    return Path(__file__).resolve().parents[3]


def _ensure_eval_importable() -> None:
    """Make the repo-local `eval/` package importable.

    It's deliberately not part of the installed `finance_qna` package (see
    design/eval-harness-plan.md §4.2), so a plain `pip install -e .` doesn't
    put the repo root on `sys.path` the way it puts `src/` there. This adds it,
    scoped to only the commands that actually need `eval.*`.
    """
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@data_app.command("generate")
def data_generate(
    seed: int = typer.Option(42, help="Random seed for reproducible generation."),
    db_path: Path | None = typer.Option(None, help="Where to write the SQLite database."),
) -> None:
    """Generate the synthetic transaction dataset."""
    path = db_path or get_settings().db_path
    generate_dataset(seed=seed, db_path=path)
    typer.echo(f"Generated dataset at {path} (seed={seed}).")


@app.command()
def ask(question: str) -> None:
    """Answer a single question and print the result, with its trace saved under runs/."""
    adapter = build_adapter(get_settings())
    run_dir = Path("runs") / uuid.uuid4().hex[:8]

    trace = adapter.run_turn(question, prior_turns=[])
    trace_path = write_trace(trace, run_dir)

    typer.echo(trace.final_answer)
    typer.echo(
        f"\n[route={trace.route} groundedness_ok={trace.groundedness_ok} trace={trace_path}]"
    )


@app.command()
def chat() -> None:
    """Start an interactive multi-turn chat session (not persisted across runs)."""
    adapter = build_adapter(get_settings())
    run_dir = Path("runs") / uuid.uuid4().hex[:8]

    typer.echo("Chat started. Type 'exit' or 'quit' to end.")
    prior_turns: list[RunTrace] = []
    while True:
        question = typer.prompt("You")
        if question.strip().lower() in {"exit", "quit"}:
            break

        trace = adapter.run_turn(question, prior_turns)
        write_trace(trace, run_dir)
        typer.echo(f"Assistant: {trace.final_answer}")
        prior_turns.append(trace)


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option("all", help="Which suite to run: single_turn, multi_turn, or all."),
    label: str = typer.Option("", help="A short label for this run, e.g. a git short-SHA."),
) -> None:
    """Run the eval test set against the configured agent and print a pass-rate summary."""
    _ensure_eval_importable()
    from eval.runner import run_eval
    from eval.schema import load_test_cases
    from eval.store import save_run

    repo_root = _repo_root()
    testset_dir = repo_root / "eval" / "testset"
    suite_files = {
        "single_turn": [testset_dir / "single_turn.yaml"],
        "multi_turn": [testset_dir / "multi_turn.yaml"],
        "all": [testset_dir / "single_turn.yaml", testset_dir / "multi_turn.yaml"],
    }
    if suite not in suite_files:
        typer.echo(f"Unknown suite {suite!r}; choose from {sorted(suite_files)}.", err=True)
        raise typer.Exit(code=1)

    cases = load_test_cases(*suite_files[suite])
    adapter = build_adapter(get_settings())

    run_id = uuid.uuid4().hex[:8]
    run_dir = repo_root / "runs" / f"eval-{run_id}"
    record = run_eval(adapter, cases, run_dir, label=label, run_id=run_id)
    save_run(record, runs_dir=repo_root / "eval" / "runs")

    typer.echo(f"Run {record.run_id} ({len(record.results)} cases, agent={record.agent_id}):")
    for metric, rate in sorted(record.summary.items()):
        typer.echo(f"  {metric}: {rate:.0%}")


@eval_app.command("dashboard")
def eval_dashboard() -> None:
    """Launch the Streamlit eval dashboard (requires `pip install -e '.[dashboard]'`)."""
    import shutil
    import subprocess

    if shutil.which("streamlit") is None:
        typer.echo(
            "streamlit isn't installed. Run: pip install -e '.[dashboard]'",
            err=True,
        )
        raise typer.Exit(code=1)

    dashboard_path = _repo_root() / "eval" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=False)


if __name__ == "__main__":
    app()
