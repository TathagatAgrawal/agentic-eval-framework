# `tracing/`

The "show its reasoning transparently" feature, and the model-/architecture-agnostic contract every `AgentAdapter` implementation returns.

## `trace.py`

- **`RunTrace`** — a complete, JSON-serializable record of one turn: `question`, `resolved_question`, `route`, the full `ledger` (every tool call and result), `draft_answer` (a `StructuredAnswer` with cited claims), `groundedness_ok`, `retries`, `final_answer`, and `latency_ms`. This is the *only* thing that crosses the `AgentAdapter` boundary (`agent/adapter.py`) — the CLI and the eval harness never see `AgentState`, `TurnMemory`, or anything LangGraph-specific, only a sequence of `RunTrace`s.
- **`build_trace(state, turn_id, latency_ms)`** — constructs a `RunTrace` from a completed LangGraph turn's final `AgentState`. Only the `langgraph` architecture uses this; the other three architectures build a `RunTrace` directly in their own `run_turn`, since they have no `AgentState` to convert from.
- **`turn_memory_from_trace(trace, turn_id)`** — the reverse direction: converts a *prior* turn's `RunTrace` into the `TurnMemory` the `contextualize` stage needs, so a caller passing `prior_turns: list[RunTrace]` into `run_turn` never has to know `AgentState`/`TurnMemory` exist.
- **`write_trace(trace, run_dir)`** / **`read_trace(path)`** — JSON persistence to `run_dir/<turn_id>.json`. `finance-qna ask`/`chat` write to `runs/<session_id>/`; the eval runner writes to `runs/eval-<run_id>/<case_id>/<turn_id>.json` in addition to the aggregated scores it saves separately via `eval/store.py`.

## Why this exists as its own module

Keeping trace construction outside the graph (built by the *caller*, from a finished state, rather than written from inside a node) keeps the LangGraph graph a pure function of state, and makes `RunTrace` the eval harness's sole way to inspect what a run actually did — question → resolved question → ledger → groundedness verdict → final answer — regardless of which of the four architectures produced it.
