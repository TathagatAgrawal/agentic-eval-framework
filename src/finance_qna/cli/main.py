"""The command-line interface: `data generate` builds the synthetic dataset,
`ask` answers one question, `chat` starts an interactive multi-turn session,
and `eval run`/`eval dashboard` drive the evaluation harness. Output is
rendered with `rich` (panels, styled text, progress bars, tables) rather than
plain `typer.echo`.

`ask`, `chat`, and `eval run` all drive the agent exclusively through
`AgentAdapter` (see `agent/adapter.py`) -- the CLI never touches LangGraph,
`AgentState`, or `TurnMemory` directly, and which model/architecture runs is
entirely a function of `Settings` (`agent_architecture` in `.env`, plus that
architecture's own config), not CLI code. Per the project's non-goals, this is
the whole interface -- no web UI, and `chat` sessions are never persisted
across process runs.
"""

import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from finance_qna.agent.adapter import build_adapter
from finance_qna.config import get_settings
from finance_qna.data.generate import generate as generate_dataset
from finance_qna.tracing.trace import RunTrace, write_trace

if TYPE_CHECKING:
    from eval.results import TestCaseResult
    from eval.schema import TestCase

app = typer.Typer(help="Personal financial statement Q&A agent.")
data_app = typer.Typer(help="Manage the synthetic transaction dataset.")
eval_app = typer.Typer(help="Run and inspect the evaluation harness.")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")

console = Console()


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


def _route_style(route: str | None) -> str:
    """Return a rich color style name for a route value."""
    return {"answer": "green", "clarify": "yellow", "refuse": "magenta"}.get(route or "", "red")


def _groundedness_text(ok: bool) -> Text:
    """Return a colored label for a groundedness verdict."""
    return Text("grounded", style="bold green") if ok else Text("unverified", style="bold yellow")


@data_app.command("generate")
def data_generate(
    seed: int = typer.Option(42, help="Random seed for reproducible generation."),
    db_path: Path | None = typer.Option(None, help="Where to write the SQLite database."),
) -> None:
    """Generate the synthetic transaction dataset."""
    path = db_path or get_settings().db_path
    generate_dataset(seed=seed, db_path=path)
    console.print(f"[green]Generated[/] dataset at [bold]{path}[/] (seed={seed}).")


@app.command()
def ask(question: str) -> None:
    """Answer a single question and print the result, with its trace saved under runs/."""
    adapter = build_adapter(get_settings())
    run_dir = Path("runs") / uuid.uuid4().hex[:8]

    trace = adapter.run_turn(question, prior_turns=[])
    trace_path = write_trace(trace, run_dir)

    route_style = _route_style(trace.route)
    subtitle = Text.assemble(
        ("route=", "dim"),
        (str(trace.route), route_style),
        ("  ", ""),
        _groundedness_text(trace.groundedness_ok),
    )
    console.print(
        Panel(
            trace.final_answer or "(no answer)",
            title=question,
            subtitle=subtitle,
            border_style=route_style,
        )
    )
    console.print(f"[dim]trace: {trace_path}[/dim]")


@app.command()
def chat() -> None:
    """Start an interactive multi-turn chat session (not persisted across runs)."""
    adapter = build_adapter(get_settings())
    run_dir = Path("runs") / uuid.uuid4().hex[:8]

    console.print(
        "[bold]Chat started.[/bold] Type [bold]exit[/bold] or [bold]quit[/bold] to end.\n"
    )
    prior_turns: list[RunTrace] = []
    while True:
        question = Prompt.ask("[bold cyan]You[/bold cyan]")
        if question.strip().lower() in {"exit", "quit"}:
            break

        trace = adapter.run_turn(question, prior_turns)
        write_trace(trace, run_dir)

        route_style = _route_style(trace.route)
        console.print(
            f"[bold green]Assistant[/bold green] "
            f"[{route_style}]({trace.route})[/{route_style}]: {trace.final_answer}"
        )
        if not trace.groundedness_ok:
            console.print(
                "  [yellow]⚠ one or more figures above could not be fully verified[/yellow]"
            )
        prior_turns.append(trace)


def _default_case_delay_seconds() -> float:
    """Read the default inter-case delay from EVAL_CASE_DELAY_SECONDS (0 if unset)."""
    return float(os.environ.get("EVAL_CASE_DELAY_SECONDS", "0"))


def _rate_style(rate: float) -> str:
    """Color a pass rate: green at/above 80%, yellow at/above 50%, red otherwise."""
    if rate >= 0.8:
        return "green"
    if rate >= 0.5:
        return "yellow"
    return "red"


def _overall_metrics(summary: dict[str, float]) -> dict[str, float]:
    """Extract the 'overall.<metric>' entries from a run's summary, unprefixed."""
    return {
        key.removeprefix("overall."): rate
        for key, rate in summary.items()
        if key.startswith("overall.")
    }


def _metrics_by_category(summary: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group the non-overall summary entries by category -> {metric: rate}."""
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for key, rate in summary.items():
        if key.startswith("overall."):
            continue
        category, _, metric = key.partition(".")
        grouped[category][metric] = rate
    return grouped


def _overall_table(overall: dict[str, float]) -> Table:
    """Build a Metric -> Pass Rate table for the overall summary."""
    table = Table(title="Overall", header_style="bold")
    table.add_column("Metric")
    table.add_column("Pass Rate", justify="right")
    for metric in sorted(overall):
        rate = overall[metric]
        table.add_row(metric, f"[{_rate_style(rate)}]{rate:.0%}[/{_rate_style(rate)}]")
    return table


def _category_table(by_category: dict[str, dict[str, float]]) -> Table:
    """Build a Category x Metric pass-rate table."""
    metrics = sorted({metric for rates in by_category.values() for metric in rates})
    table = Table(title="By Category", header_style="bold")
    table.add_column("Category")
    for metric in metrics:
        table.add_column(metric, justify="right")
    for category in sorted(by_category):
        row = [category]
        for metric in metrics:
            rate = by_category[category].get(metric)
            row.append(
                f"[{_rate_style(rate)}]{rate:.0%}[/{_rate_style(rate)}]"
                if rate is not None
                else "[dim]—[/dim]"
            )
        table.add_row(*row)
    return table


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option("all", help="Which suite to run: single_turn, multi_turn, or all."),
    label: str = typer.Option("", help="A short label for this run, e.g. a git short-SHA."),
    limit: int | None = typer.Option(
        None, help="Only run the first N cases (useful under a tight rate limit)."
    ),
    delay: float | None = typer.Option(
        None,
        help="Seconds to sleep between cases (not before the first), to stay under a "
        "provider's per-minute rate limit. Defaults to EVAL_CASE_DELAY_SECONDS (0 = "
        "disabled). Pass 0 to force-disable regardless of that env var.",
    ),
) -> None:
    """Run the eval test set against the configured agent and print a pass-rate report."""
    _ensure_eval_importable()
    from eval.runner import run_eval
    from eval.schema import load_test_cases

    repo_root = _repo_root()
    testset_dir = repo_root / "eval" / "testset"
    suite_files = {
        "single_turn": [testset_dir / "single_turn.yaml"],
        "multi_turn": [testset_dir / "multi_turn.yaml"],
        "all": [testset_dir / "single_turn.yaml", testset_dir / "multi_turn.yaml"],
    }
    if suite not in suite_files:
        console.print(f"[red]Unknown suite {suite!r}; choose from {sorted(suite_files)}.[/red]")
        raise typer.Exit(code=1)

    cases = load_test_cases(*suite_files[suite])
    if limit is not None:
        cases = cases[:limit]
    adapter = build_adapter(get_settings())
    delay_seconds = _default_case_delay_seconds() if delay is None else delay

    run_id = uuid.uuid4().hex[:8]
    run_dir = repo_root / "runs" / f"eval-{run_id}"
    eval_runs_dir = repo_root / "eval" / "runs"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting eval run...", total=len(cases))

        def on_case_start(case: "TestCase", index: int, total: int) -> None:
            """Update the progress bar's description to the case now running."""
            progress.update(task_id, description=f"Running {case.id}")

        def on_case_done(case: "TestCase", result: "TestCaseResult") -> None:
            """Advance the progress bar by one completed case."""
            progress.advance(task_id)

        record = run_eval(
            adapter,
            cases,
            run_dir,
            label=label,
            run_id=run_id,
            runs_dir=eval_runs_dir,
            delay_seconds=delay_seconds,
            on_case_start=on_case_start,
            on_case_done=on_case_done,
        )

    console.print()
    console.print(
        f"[bold]Run {record.run_id}[/bold] ({len(record.results)} cases, agent={record.agent_id})"
    )

    errored = [r for r in record.results if r.error]
    if errored:
        error_table = Table(
            title=f"{len(errored)} case(s) did not complete", header_style="bold red"
        )
        error_table.add_column("Case ID")
        error_table.add_column("Error")
        for r in errored:
            error_table.add_row(r.case_id, r.error or "")
        console.print(error_table)

    console.print(_overall_table(_overall_metrics(record.summary)))
    console.print(_category_table(_metrics_by_category(record.summary)))


@eval_app.command("dashboard")
def eval_dashboard() -> None:
    """Launch the Streamlit eval dashboard (requires `pip install -e '.[dashboard]'`)."""
    import shutil
    import subprocess

    if shutil.which("streamlit") is None:
        console.print("[red]streamlit isn't installed.[/red] Run: pip install -e '.[dashboard]'")
        raise typer.Exit(code=1)

    dashboard_path = _repo_root() / "eval" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=False)


if __name__ == "__main__":
    app()
