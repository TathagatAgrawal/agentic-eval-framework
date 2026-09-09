# `eval/`

An architecture-agnostic evaluation harness: a hand-curated test set with mechanically-derived labels, six rule-based scorers (no LLM-as-judge), a resilient runner, and a flat-file regression store. Scores any `AgentAdapter` (see `src/finance_qna/agent/README.md`) identically, through the same `run_turn(question, prior_turns) -> RunTrace` interface the CLI uses.

Deliberately **not** part of the installed `finance_qna` package — see "Why `eval/` lives outside `src/`" below.

## Layout

```
eval/
├── schema.py         # TestCase/ExpectedResult/Turn + load_test_cases()  (input schema)
├── testset/
│   ├── single_turn.yaml
│   └── multi_turn.yaml
├── scorers/          # one file per metric, each a pure function of (RunTrace, ExpectedResult)
├── runner.py         # run_test_case / run_eval: drives an AgentAdapter and scores it
├── results.py        # TestCaseResult / RunRecord  (output schema)
├── store.py          # save_run / load_all_runs — flat JSON files under eval/runs/
├── runs/             # persisted RunRecords, one <run_id>.json per eval run
└── dashboard.py       # Streamlit dashboard for browsing eval/runs/ history
```

## Test case schema (`schema.py`)

A `TestCase` is either single-turn (`question` + `expected`) or a multi-turn sequence (`turns`, each with its own `expected`) — never both, enforced by a validator. `ExpectedResult` declares: `behavior` (`answer`/`clarify`/`refuse`), `value`/`values` + `tolerance` (for `answer` cases), `expected_tools` (tool names that should appear in the ledger), `expected_context` (a partial-match check on ledger args, for verifying multi-turn reference resolution), `optimal_tool_calls` (the efficiency baseline, see below), and `groundedness_ok`.

**Every expected value is computed by running the real tool functions against the seeded fixture database, never hand-computed** — this is enforced by convention (see the header comments in the YAML files), and was independently re-verified for the current test set by calling `tools/query_tools.py`, `compare_tools.py`, and `trend_tools.py` directly against `data/synthetic_transactions.db`, bypassing every architecture and the LLM entirely.

`load_test_cases(*paths)` loads and validates every case from one or more YAML files, raising on a duplicate id across files.

## Scorers (`scorers/`)

Each scorer is a pure function `(RunTrace, ExpectedResult) -> ScoreResult`, returning `passed: bool | None` — `None` means "not applicable" (e.g. a clarify/refuse turn has no numeric claims to check) and is excluded from pass-rate aggregation rather than counted as a pass or fail:

- **`numeric_correctness`** — does any claimed value match the expected value (within tolerance)?
- **`groundedness`** — scored two independent ways: the agent's own self-reported `groundedness_ok`, *and* an independent recomputation by re-running `agent.groundedness.verify()` outside the agent, against the raw ledger and claims. The second exists specifically so groundedness is measured the same way regardless of whether an architecture implements its own internal check at all — the harness never simply trusts what the system under test says about itself.
- **`contextual_correctness`** — for multi-turn follow-ups with a declared `expected_context`: did the resolved category/date range/comparison baseline actually match what the follow-up implied? Checked by flattening ledger args and partial-matching.
- **`clarification`** / **`refusal`** — did the agent ask/decline when (and only when) it should have, checked in both directions.
- **`efficiency`** — actual tool calls vs. the case's labeled `optimal_tool_calls`; reports `overage` (actual − optimal) as a magnitude, not just a pass/fail bit.

Every scorer runs unconditionally for every turn (`runner.score_turn`); the runner never special-cases by expected behavior, since each scorer already knows when it doesn't apply.

## The efficiency baseline

Each case is labeled with the fewest tool calls achievable with the *current* toolset (e.g. one `aggregate_spending_tool` call for a single-category single-month total; one `compare_periods_tool` call for a two-period comparison, not two separate aggregations) — verified by hand against the actual tool signatures, not assumed.

## Runner (`runner.py`)

`run_test_case(adapter, case, run_dir)` drives one case (single- or multi-turn) and scores it turn-by-turn, aggregating a multi-turn case's per-metric result as "pass only if every turn where that metric applied passed." If a turn raises (e.g. a provider rate limit), the case stops there but every turn that *did* complete is still scored and returned, with `error` set to what stopped it — no completed work is discarded.

`run_eval(adapter, cases, run_dir, ...)` drives every case, and — if given `runs_dir` — **checkpoints the full `RunRecord` to disk after every single case**, not just once at the end, so an interrupted run never loses the cases that already completed. Accepts `delay_seconds` (sleep between cases, for a per-minute rate limit) and `on_case_start`/`on_case_done` callbacks (used by the CLI to drive a `rich` progress bar) — the runner itself has no UI-library dependency.

## Regression store (`store.py`)

Flat, git-trackable JSON files under `eval/runs/` (one `<run_id>.json` per run) rather than a database — the point is that run-to-run history stays directly diffable in a PR. `save_run`/`load_all_runs` are the only two operations.

## Dashboard (`dashboard.py`)

A Streamlit app (`finance-qna eval dashboard`, requires `pip install -e ".[dashboard]"`) for browsing `eval/runs/` history — pass rates over time, per-category breakdowns, per-architecture comparison.

## Why `eval/` lives outside `src/`

The evaluation harness is a tool for *judging* the agent, not a dependency of it — an architecture must never be able to import from `eval/`, so nothing under `src/finance_qna/` ever does. `cli/main.py`'s `eval` subcommands are the one deliberate, scoped exception that reaches across this boundary (see `src/finance_qna/cli/README.md`), inserting the repo root onto `sys.path` only for those commands.

## Running

```bash
finance-qna eval run                              # all 15 cases, current AGENT_ARCHITECTURE
finance-qna eval run --suite single_turn --limit 5 --delay 5
finance-qna eval dashboard
```

## Adding a test case

Add a case to `testset/single_turn.yaml` or `testset/multi_turn.yaml` with a unique `id`, computing its `expected.value`/`values` and `optimal_tool_calls` by actually calling the relevant tool function(s) against the live database — never by hand-computing the number. No runner or scorer code needs to change.
