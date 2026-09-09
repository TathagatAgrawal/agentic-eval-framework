# Agentic Eval Framework

An architecture-agnostic evaluation framework for agentic LLM systems: score any agent on objective, rule-based metrics — numeric correctness, groundedness (self-reported *and* independently recomputed), contextual correctness, clarification/refusal behavior, and tool-call efficiency — with no LLM-as-judge anywhere in the loop. The only thing the framework requires of an agent is one interface, `AgentAdapter.run_turn(question, prior_turns) -> RunTrace` (see [eval/README.md](eval/README.md) and [src/finance_qna/agent/README.md](src/finance_qna/agent/README.md)); anything that implements it can be scored, compared, and tracked across runs.

The framework is demonstrated here on a concrete reference task — a personal financial Q&A agent — implemented as **four interchangeable architectures** (a LangGraph state machine, a monolithic ReAct loop, a context-stuffing design, and a plan-and-execute design) so the framework itself has something non-trivial to compare. See [report/report.md](report/report.md) for the full four-way comparison this framework produced, and [design/eval-harness-plan.md](design/eval-harness-plan.md) / [design/alternative-architectures-plan.md](design/alternative-architectures-plan.md) for the design behind it. [design/project-overview.md](design/project-overview.md) and [design/high-level-design.md](design/high-level-design.md) / [design/low-level-design.md](design/low-level-design.md) cover the reference agent itself.

Built with Python; the reference agent uses [LangGraph](https://github.com/langchain-ai/langgraph)/[LangChain](https://github.com/langchain-ai/langchain) and Google Gemini, but the eval framework has no dependency on either — it only knows about `AgentAdapter` and `RunTrace`. The reference agent's dataset is entirely synthetic and deterministically generated; no real financial data is used.

## Setup

**Requirements:** Python 3.11+, a [Google AI Studio API key](https://aistudio.google.com/apikey) (only needed to run the reference agent — the eval framework itself has no model dependency).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the project with dev dependencies (add the `dashboard` extra for `eval dashboard`)
pip install -e ".[dev]"
# pip install -e ".[dev,dashboard]"

# 3. Configure your Gemini API key
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=...

# 4. Generate the synthetic transaction dataset
finance-qna data generate
```

`data generate` writes `data/synthetic_transactions.db` (SQLite) and `data/ground_truth.json` (the computed aggregates and labeled events used as ground truth). It's deterministic — the same `--seed` (default `42`) always produces the same dataset.

## Evaluation Framework

```bash
# Run the full 15-case test set against whichever architecture AGENT_ARCHITECTURE selects
finance-qna eval run --label my-run

# Run just one suite, cap the number of cases, or throttle for a rate limit
finance-qna eval run --suite single_turn --limit 5 --delay 5

# Browse past runs (requires the `dashboard` extra)
finance-qna eval dashboard
```

Results are saved to `eval/runs/<run_id>.json` (checkpointed after every case, so a rate-limit interruption never loses completed work) and a `rich`-rendered pass-rate report prints to the console. See [eval/README.md](eval/README.md) for the scorers, test-case schema, and how to add a new test case or wire up a new agent for comparison.

The framework scores six metrics per turn, every one a deterministic function of the turn's execution trace — no metric depends on a second LLM's judgment:

- **Numeric correctness** — does any claimed value match the expected value within tolerance?
- **Groundedness**, computed two independent ways: self-reported (does the agent's own check agree with the expected verdict?) and harness-computed (the same verification re-run independently outside the agent, so groundedness is never just taken on the agent's word).
- **Contextual correctness** — in a multi-turn sequence, did the resolved category/date range/comparison baseline actually match what the follow-up implied?
- **Clarification** / **Refusal** — did the agent ask when (and only when) it should have, and decline out-of-scope questions symmetrically?
- **Efficiency** — actual tool calls versus a hand-verified *optimal* count, reporting the overage as a magnitude, not just pass/fail.

### Scoring a new agent

Any agent can be evaluated by this framework — it never touches LangGraph, Gemini, or anything else specific to the reference implementation. Implement `AgentAdapter` (an `id: str` attribute and `run_turn(question, prior_turns) -> RunTrace`, see `src/finance_qna/agent/adapter.py` for the protocol and `src/finance_qna/tracing/trace.py` for `RunTrace`), point the harness at it, and the same test set, scorers, runner, and regression store apply unchanged.

## The Reference Agent

The task the framework is demonstrated on: answering natural-language questions about a user's own synthetic transaction history — aggregations, comparisons, trend/anomaly detection, and multi-turn follow-ups — while declining out-of-scope questions and asking for clarification rather than guessing.

```bash
# Ask a single question
finance-qna ask "How much did I spend on groceries in July 2024?"

# Start an interactive multi-turn session
finance-qna chat
```

Example `chat` session, showing multi-turn reference resolution:

```
You: How much did I spend on dining in Q1 2025?
Assistant: In Q1 2025, you spent $1,325.97 across 63 transactions on dining.
You: What about groceries?
Assistant: You spent $1,613.01 on groceries across 27 transactions in Q1 2025.
You: Is that more or less than dining?
Assistant: The amount spent on groceries in Q1 2025 was $1,613.01, which is more than dining ($1,325.97).
```

Every turn's full reasoning trace — the resolved question, every tool call and result, the groundedness verdict, and the final answer — is written to `runs/<session_id>/<turn_id>.json` for inspection.

### Inspecting the data

The dataset is a plain SQLite file, so any SQLite client works:

```bash
sqlite3 data/synthetic_transactions.db

sqlite> .tables
accounts  categories  merchants  transactions

sqlite> .schema transactions

sqlite> SELECT date, amount, description FROM transactions ORDER BY date DESC LIMIT 10;
```

`data/ground_truth.json` is plain JSON and has the same information pre-aggregated — monthly totals per category, and the two labeled events (a subscription price change and a category spend spike) that the trend/anomaly tools are built to detect:

```bash
cat data/ground_truth.json | python -m json.tool | less
```

A DB browser GUI (e.g. [DB Browser for SQLite](https://sqlitebrowser.org/)) also opens `data/synthetic_transactions.db` directly if you'd rather click through the tables.

### Four architectures, one interface

`AGENT_ARCHITECTURE` (in `.env`) selects which implementation `finance-qna ask`/`chat`/`eval run` drives — all four answer the same questions against the same tools and database, behind the identical `AgentAdapter` interface (see [src/finance_qna/agent/README.md](src/finance_qna/agent/README.md)):

| Value | Architecture | Summary |
|---|---|---|
| `langgraph` (default) | State machine | Explicit contextualize/route/act-tool_node-loop/draft/ground_check stages, built with LangGraph. |
| `monolithic` | Monolithic ReAct | One system prompt, one tool-calling loop, one final call that routes and drafts together. |
| `context_stuffing` | Context-stuffing | No tool calls; a precomputed summary of the whole dataset is injected into the draft prompt every turn. |
| `plan_execute` | Plan-and-execute | Plans every needed tool call upfront in one batch, executes without interleaving, re-plans (not re-loops) if the draft is ungrounded. |

None of the four is a strict winner — see [report/report.md](report/report.md) for the full, metric-by-metric comparison this framework produced, and the specific failure traces behind it.

These four architectures exist specifically to give the eval framework something to compare — a framework with only one implementation to score can't demonstrate that its metrics actually discriminate between designs.

## Configuration

Settings are read from `.env` (see `.env.example`). `src/finance_qna/config.py` holds only what every architecture needs; each architecture has its own `<PREFIX>_*` settings, so switching `AGENT_ARCHITECTURE` never leaves stale, irrelevant config lying around:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | required; your Gemini API key |
| `AGENT_ARCHITECTURE` | `langgraph` | which `AgentAdapter` to build: `langgraph`, `monolithic`, `context_stuffing`, or `plan_execute` |
| `DB_PATH` | `data/synthetic_transactions.db` | path to the SQLite dataset |
| `EVAL_CASE_DELAY_SECONDS` | `0` | seconds to sleep between eval cases (see `eval run --delay`) |
| `LANGGRAPH_AGENT_MODEL` / `LANGGRAPH_CLASSIFIER_MODEL` | `gemini-3.5-flash-lite` | models used by the `langgraph` architecture |
| `LANGGRAPH_MAX_TOOL_STEPS` | `6` | cap on tool calls per turn (`langgraph`) |
| `LANGGRAPH_GROUNDEDNESS_RETRY_LIMIT` | `1` | draft retries on a failed groundedness check (`langgraph`) |
| `MONOLITHIC_MODEL` / `MONOLITHIC_MAX_TOOL_STEPS` | `gemini-3.5-flash-lite` / `6` | model and tool-step cap for the `monolithic` architecture |
| `CONTEXT_STUFFING_MODEL` | `gemini-3.5-flash-lite` | model for the `context_stuffing` architecture |
| `PLAN_EXECUTE_MODEL` / `PLAN_EXECUTE_GROUNDEDNESS_RETRY_LIMIT` | `gemini-3.5-flash-lite` / `1` | model and re-plan limit for the `plan_execute` architecture |

See `.env.example` for the full, commented list.

## Development

```bash
ruff format .          # format
ruff check . --fix     # lint
mypy src eval           # type-check (strict, scoped to src/finance_qna and eval)
pytest -q               # run the test suite
```

The test suite never calls the real Gemini API by default — LLM calls are mocked (see `tests/fakes.py`). One opt-in live test confirms real API connectivity:

```bash
RUN_LIVE_TESTS=1 pytest tests/test_config.py
```

## Project structure

```
eval/               # the eval framework itself: schema, scorers, runner, regression store, dashboard → eval/README.md
report/             # the four-architecture comparison report this framework produced

src/finance_qna/    # the reference agent evaluated by eval/
├── config.py       # universal Settings + Gemini client factory
├── data/           # schema, DB engine, synthetic data generator          → src/finance_qna/data/README.md
├── tools/          # the agent's only path to the data: typed, validated query/compare/trend tools → src/finance_qna/tools/README.md
├── agent/          # the four AgentAdapter implementations + shared groundedness/answer/prompt code → src/finance_qna/agent/README.md
├── tracing/        # per-turn RunTrace JSON logging                       → src/finance_qna/tracing/README.md
└── cli/            # `finance-qna` command-line entry point               → src/finance_qna/cli/README.md

design/             # the design docs referenced throughout this README
```

Each subdirectory listed above has its own `README.md` with the detail specific to that module — start at the top-level one linked from a section above for whichever part you're touching.
