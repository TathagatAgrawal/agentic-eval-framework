"""The command-line interface: `data generate` builds the synthetic dataset,
`ask` answers one question, and `chat` starts an interactive multi-turn session.

Both `ask` and `chat` drive the agent exclusively through `AgentAdapter` (see
`agent/adapter.py`) -- the CLI never touches LangGraph, `AgentState`, or
`TurnMemory` directly, and which model/architecture runs is entirely a function
of `Settings` (`agent_model`/`agent_architecture` in `.env`), not CLI code. Per
the project's non-goals, this is the whole interface -- no web UI, and `chat`
sessions are never persisted across process runs.
"""

import uuid
from pathlib import Path

import typer

from finance_qna.agent.adapter import build_adapter
from finance_qna.config import get_settings
from finance_qna.data.generate import generate as generate_dataset
from finance_qna.tracing.trace import RunTrace, write_trace

app = typer.Typer(help="Personal financial statement Q&A agent.")
data_app = typer.Typer(help="Manage the synthetic transaction dataset.")
app.add_typer(data_app, name="data")


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


if __name__ == "__main__":
    app()
