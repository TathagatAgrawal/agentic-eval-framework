# `agent/`

Four interchangeable agent architectures, all reachable through one interface, plus the pieces they share.

## The shared interface

**`adapter.py`** defines the `AgentAdapter` protocol — `run_turn(question, prior_turns: list[RunTrace]) -> RunTrace` — the single seam the CLI (`cli/main.py`) and the eval harness (`eval/runner.py`) drive an agent through. Neither caller needs to know LangGraph or Gemini exist; they only ever call `run_turn` and exchange `RunTrace`s (`tracing/trace.py`). `build_adapter(settings)` is the one place that picks a concrete implementation based on `settings.agent_architecture` — adding a fifth architecture means adding a branch here, not touching the CLI or the eval runner.

## The four architectures

| `AGENT_ARCHITECTURE` | File | Config (env prefix) |
|---|---|---|
| `langgraph` (default) | `langgraph_adapter.py` | `LANGGRAPH_*` |
| `monolithic` | `monolithic_adapter.py` | `MONOLITHIC_*` |
| `context_stuffing` | `context_stuffing_adapter.py` | `CONTEXT_STUFFING_*` |
| `plan_execute` | `plan_execute_adapter.py` | `PLAN_EXECUTE_*` |

See [design/alternative-architectures-plan.md](../../../design/alternative-architectures-plan.md) for the rationale behind each, and [report/report.md](../../../report/report.md) for how they actually compare on the shared eval set.

**LangGraph** (`langgraph_adapter.py`, `graph.py`, `nodes.py`, `state.py`, `route.py`, `contextualize.py`) — a `StateGraph` with explicit stages: `contextualize` (resolve references against prior turns), `route` (answer/clarify/refuse), an `act`/`tool_node` loop that interleaves tool calls with reasoning, `draft_answer` (cites claims to ledger entries), and `ground_check` (retries the draft, then falls back to a caveat if retries are exhausted). This is the only architecture with its own graph — the other three call the shared prompts/stages directly as plain functions/LLM calls, without a graph runtime.

**Monolithic ReAct** (`monolithic_adapter.py`) — one system prompt (`MONOLITHIC_SYSTEM_PROMPT`), one ReAct tool-calling loop (`_run_tool_loop`), then one final structured-output call (`MonolithicResult`) that decides routing and drafts the answer together. No separate contextualize/route stage, no retry on an ungrounded draft — a caveat is applied immediately instead.

**Context-Stuffing** (`context_stuffing_adapter.py`) — no tool calls. `compute_context_table(engine)` precomputes a monthly-totals-per-category summary plus every detected subscription change, once per adapter instance, and every turn injects that whole table into the draft prompt. `contextualize`/`route`/`clarify`/`refuse` are reused unchanged from the LangGraph baseline's prompts. The one ledger entry this architecture ever produces is synthetic — the context table wrapped as a single `tool_name="context_table"` entry — purely so the shared `groundedness.verify()` can check it without special-casing. This is also its known structural weakness: with one giant ledger entry instead of scoped tool results, the groundedness check has very little to actually falsify (see the report's discussion section).

**Plan-and-Execute** (`plan_execute_adapter.py`) — `contextualize`/`route` reused unchanged again; the tool-use stage is replaced with an upfront batch plan (`_plan_and_execute`: one `llm.bind_tools(...)` call that may request several tool calls at once, all executed before the model is consulted again — no interleaving) and, if the draft is ungrounded, a bounded re-plan against the specific unsupported claims rather than a full retry from scratch. The planning step deliberately uses the same native function-calling every other architecture uses for execution, not a custom structured-output `Plan` schema — an earlier version tried the latter and Gemini reliably returned an empty plan for it, traced to the open-ended `args: dict` field being a poor target for constrained JSON decoding (see the module docstring for the full account).

## Shared building blocks

- **`answer.py`** — `StructuredAnswer`/`Claim`: the universal draft-answer schema every architecture's draft stage produces, citing a `ledger_id` (and optionally a description of a derived computation) for every numeric claim.
- **`groundedness.py`** — `verify()`: a deterministic, LLM-free function checking each `Claim` against the ledger's values (direct match, or a simple derived sum/diff/pct-change across ledger values). Used identically by every architecture's own groundedness check *and* independently re-run by the eval harness's `harness_groundedness_ok` scorer, so groundedness is never measured only by trusting what the system under test says about itself.
- **`state.py`** — `LedgerEntry` (the record of one tool call: name, args, result, timestamp) and `TurnMemory` (what a prior turn contributes to `contextualize`'s prompt); `AgentState` for the LangGraph graph specifically.
- **`prompts.py`** — every LLM-facing prompt template, including the ones shared across architectures (`CONTEXTUALIZE_INSTRUCTIONS`, `ROUTE_INSTRUCTIONS`, `CLARIFY_INSTRUCTIONS`, `REFUSE_INSTRUCTIONS`, `DRAFT_ANSWER_INSTRUCTIONS`, `GROUNDEDNESS_CAVEAT`) and the ones specific to one architecture (`MONOLITHIC_*`, `PLAN_INSTRUCTIONS`/`REPLAN_INSTRUCTIONS`).
- **`contextualize.py`** / **`route.py`** — `ContextualizeResult`/`RouteDecision`, the structured-output schemas the shared contextualize/route stage produces.
- **`nodes.py`** — the LangGraph node functions, plus a few free functions reused by the non-graph architectures (`extract_text`, `format_session_memory`, `serialize_tool_result`).

## Adding a fifth architecture

Implement `AgentAdapter` (an `id: str` attribute and `run_turn`), give it its own `BaseSettings` subclass with a distinct `env_prefix` if it needs configuration, and add a branch to `build_adapter` in `adapter.py`. Reuse `groundedness.verify()`, `answer.StructuredAnswer`, and whichever of the shared `prompts.py` stages fit, rather than reimplementing them — that's what let Context-Stuffing and Plan-and-Execute both reuse `contextualize`/`route` unchanged. No change to the CLI or `eval/` is needed; both already dispatch purely on `settings.agent_architecture`.
