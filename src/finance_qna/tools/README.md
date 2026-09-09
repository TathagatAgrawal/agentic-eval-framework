# `tools/`

The agent's only path to the transaction database. Every function here is plain Python taking an `Engine` as its first argument — fully unit-testable with zero LangChain or LLM involvement — and is wrapped as a LangChain tool only at the boundary where an architecture needs to bind it to a model.

## Files

- **`models.py`** — every tool argument and result as a Pydantic v2 model: `DateRange`, `TransactionFilter`, `TransactionRow`, `AggregateResult`, `CompareResult`, `TrendPoint`/`TrendResult`, `SubscriptionChangeEvent`. Because every tool call and result is one of these typed models rather than free text, a tool call is a normalized, comparable object (name + validated args) and a result can be inspected field-by-field — this is what makes the eval harness's `contextual_correctness` and groundedness scorers (`eval/scorers/`) possible at all.
- **`query_tools.py`** — `list_categories`, `get_transactions`, `aggregate_spending` (the primary "how much did I spend on X" tool; supports `group_by="category"|"month"`). Also `validate_category` and `filter_conditions`/`joined_transactions`, shared by `trend_tools.py`. Every query is built with SQLAlchemy's `select()` against validated Pydantic args — never string concatenation — so SQL injection isn't possible regardless of what ends up in a free-text field.
- **`compare_tools.py`** — `compare_periods`: total spend in two date ranges, optionally scoped to one category, plus the diff and percent change.
- **`trend_tools.py`** — `detect_trend` (a time series bucketed by month or quarter, plus its overall direction) and `detect_subscription_changes` (any two consecutive same-merchant subscription transactions with different amounts). Both compute their results server-side — SQL grouping plus Python aggregation — rather than handing raw transactions to the LLM to reason about, so the arithmetic the eval harness checks exactly is never in the model's hands.
- **`errors.py`** — `CategoryNotFoundError`, raised (with the list of valid categories) when a filter names a category that doesn't exist, so a bad LLM-supplied argument becomes a structured error a tool loop can react to, not an unhandled exception.
- **`registry.py`** — `build_tools(engine) -> list[BaseTool]`: binds every function above to one `Engine` via closures and wraps each as a LangChain `@tool`, so the LLM only ever sees the arguments it actually supplies (never the engine). This is the one place all six tools are assembled into the list every architecture's `llm.bind_tools(...)` call uses.

## Adding a tool

Add the plain function (taking `engine` first) to the appropriate `*_tools.py` file, add any new argument/result models to `models.py`, then add a `@tool`-wrapped closure for it in `registry.py`'s `build_tools`. No architecture-specific code needs to change — every `AgentAdapter` implementation gets the new tool automatically through `build_tools`.
