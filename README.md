# Personal Financial Statement Q&A Agent

An agentic system that answers natural-language questions about synthetic personal transaction data — spending by category, comparisons, trends, subscription price changes — grounded in real query results and transparent about how it got each number. See [design/project-overview.md](design/project-overview.md), [design/high-level-design.md](design/high-level-design.md), and [design/low-level-design.md](design/low-level-design.md) for the full design.

Built with Python, [LangGraph](https://github.com/langchain-ai/langgraph)/[LangChain](https://github.com/langchain-ai/langchain), and Google Gemini. No real financial data is used — the dataset is synthetic and deterministically generated.

## Setup

**Requirements:** Python 3.11+, a [Google AI Studio API key](https://aistudio.google.com/apikey).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the project with dev dependencies
pip install -e ".[dev]"

# 3. Configure your Gemini API key
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=...

# 4. Generate the synthetic transaction dataset
finance-qna data generate
```

`data generate` writes `data/synthetic_transactions.db` (SQLite) and `data/ground_truth.json` (the computed aggregates and labeled events used as ground truth). It's deterministic — the same `--seed` (default `42`) always produces the same dataset.

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

## Running the agent

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

## Configuration

Settings are read from `.env` (see `.env.example`) via `finance_qna.config.Settings`:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | required; your Gemini API key |
| `AGENT_MODEL` | `gemini-3.5-flash-lite` | model used for the reasoning/tool-calling loop |
| `CLASSIFIER_MODEL` | `gemini-3.5-flash-lite` | model used for routing/contextualizing |
| `DB_PATH` | `data/synthetic_transactions.db` | path to the SQLite dataset |
| `MAX_TOOL_STEPS` | `6` | cap on tool calls per turn |
| `GROUNDEDNESS_RETRY_LIMIT` | `1` | retries allowed if a draft answer fails the groundedness check |

## Development

```bash
ruff format .          # format
ruff check . --fix     # lint
mypy src                # type-check (strict, scoped to src/finance_qna)
pytest -q               # run the test suite
```

The test suite never calls the real Gemini API by default — LLM calls are mocked (see `tests/fakes.py`). One opt-in live test confirms real API connectivity:

```bash
RUN_LIVE_TESTS=1 pytest tests/test_config.py
```

## Project structure

```
src/finance_qna/
├── config.py       # Settings + Gemini client factory
├── data/           # schema, DB engine, synthetic data generator
├── tools/          # the agent's only path to the data: typed, validated query/compare/trend tools
├── agent/          # the LangGraph state machine (contextualize/route/act/tool_node/draft_answer/ground_check)
├── memory/         # in-process, single-session conversational memory
├── tracing/        # per-turn RunTrace JSON logging
└── cli/            # `finance-qna` command-line entry point
```

An evaluation harness (`eval/`) that scores the agent on numeric correctness, groundedness, contextual correctness, clarification/refusal behavior, and efficiency is planned but not yet built — see the project overview's stated build order (agent first, harness second).
