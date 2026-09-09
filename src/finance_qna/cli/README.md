# `cli/`

The `finance-qna` command-line entry point — the whole interface to this project; there is no web UI or persisted server process.

## `main.py`

- **`data generate`** — wraps `data.generate.generate`.
- **`ask "<question>"`** / **`chat`** — drive the agent built by `agent.adapter.build_adapter(get_settings())` (so which of the four architectures runs is entirely a function of `AGENT_ARCHITECTURE` in `.env`, never CLI code) through `AgentAdapter.run_turn`, exactly as the eval harness does. `chat` holds one `list[RunTrace]` of prior turns in-process for the session's lifetime; nothing is persisted across process runs beyond the trace files written to `runs/`.
- **`eval run`** / **`eval dashboard`** — thin wrappers around `eval.runner.run_eval` and the Streamlit dashboard (`eval/dashboard.py`). See `_ensure_eval_importable()` below.

Output is rendered with `rich` — `Panel` for single answers, styled `Text` for routes/groundedness, `Progress` (spinner + bar + count + elapsed time) during an eval run, and `Table` for the resulting pass-rate summary — rather than plain `typer.echo`.

## The one deliberate exception to "CLI never reaches outside the installed package"

`eval/` is intentionally not part of the installed `finance_qna` package (see `eval/README.md`), so a plain `pip install -e .` doesn't put the repo root on `sys.path` the way it puts `src/` there. `_ensure_eval_importable()` inserts the repo root onto `sys.path` at call time, scoped to only the `eval` subcommands, immediately before `from eval.runner import run_eval` / `from eval.schema import load_test_cases`. Every other command (`ask`, `chat`, `data generate`) never reaches outside the installed package.

## Adding a CLI option

`eval run`'s `--limit`/`--delay` options exist specifically to make live runs practical under a tight per-minute provider rate limit (`--limit N` caps the case count; `--delay` sleeps between cases and defaults to `EVAL_CASE_DELAY_SECONDS`, with `--delay 0` force-disabling it). Follow that pattern — an explicit flag that falls back to an env var — for anything else that needs to be tunable per-invocation without becoming a permanent setting.
