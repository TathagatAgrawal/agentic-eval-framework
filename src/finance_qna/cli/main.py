"""The command-line interface: `data generate` builds the synthetic dataset,
`ask` answers one question, and `chat` starts an interactive multi-turn session.

Per the project's non-goals, this is the whole interface -- no web UI, and
`chat` sessions are never persisted across process runs.
"""

import time
import uuid
from pathlib import Path
from typing import Any

import typer

from finance_qna.agent.graph import build_graph
from finance_qna.agent.state import AgentState, TurnMemory, initial_state, turn_memory_from_state
from finance_qna.config import get_llm, get_settings
from finance_qna.data.db import get_engine
from finance_qna.data.generate import generate as generate_dataset
from finance_qna.memory.session import SessionMemory
from finance_qna.tracing.trace import build_trace, write_trace

app = typer.Typer(help="Personal financial statement Q&A agent.")
data_app = typer.Typer(help="Manage the synthetic transaction dataset.")
app.add_typer(data_app, name="data")


def _build_graph() -> Any:
    """Construct the compiled agent graph from the configured settings."""
    settings = get_settings()
    engine = get_engine(settings.db_path)
    llm = get_llm(settings.agent_model)
    return build_graph(
        engine,
        llm,
        max_tool_steps=settings.max_tool_steps,
        groundedness_retry_limit=settings.groundedness_retry_limit,
    )


def _run_turn(
    graph: Any,
    question: str,
    session_memory: list[TurnMemory],
    run_dir: Path,
    turn_id: str,
) -> AgentState:
    """Invoke the graph for one turn, write its trace, and return the final state."""
    state = initial_state(question, session_memory=session_memory)
    start = time.monotonic()
    result: AgentState = graph.invoke(state)
    latency_ms = int((time.monotonic() - start) * 1000)

    trace = build_trace(result, turn_id=turn_id, latency_ms=latency_ms)
    write_trace(trace, run_dir)

    return result


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
    graph = _build_graph()
    session_id = uuid.uuid4().hex[:8]
    run_dir = Path("runs") / session_id

    result = _run_turn(graph, question, session_memory=[], run_dir=run_dir, turn_id="1")

    typer.echo(result["final_answer"])
    typer.echo(
        f"\n[route={result['route']} groundedness_ok={result['groundedness_ok']} "
        f"trace={run_dir / '1.json'}]"
    )


@app.command()
def chat() -> None:
    """Start an interactive multi-turn chat session (not persisted across runs)."""
    graph = _build_graph()
    session = SessionMemory()
    session_id = uuid.uuid4().hex[:8]
    run_dir = Path("runs") / session_id

    typer.echo("Chat started. Type 'exit' or 'quit' to end.")
    turn_number = 0
    while True:
        question = typer.prompt("You")
        if question.strip().lower() in {"exit", "quit"}:
            break
        turn_number += 1

        result = _run_turn(
            graph, question, session.recent(), run_dir=run_dir, turn_id=str(turn_number)
        )
        typer.echo(f"Assistant: {result['final_answer']}")

        if result["route"] == "answer":
            session.append(turn_memory_from_state(result))


if __name__ == "__main__":
    app()
