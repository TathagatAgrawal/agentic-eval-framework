"""Streamlit dashboard for the eval harness's regression store.

Run via `finance-qna eval dashboard`, or directly with
`streamlit run eval/dashboard.py`. Reads only `eval.store.load_all_runs()` --
never touches the agent, LangGraph, or Gemini, per
design/eval-harness-plan.md §9.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from eval.results import RunRecord, TestCaseResult  # noqa: E402
from eval.store import DEFAULT_RUNS_DIR, load_all_runs  # noqa: E402
from finance_qna.tracing.trace import read_trace  # noqa: E402


def _overall_metrics(record: RunRecord) -> dict[str, float]:
    """Extract the 'overall.<metric>' entries from a run's summary, unprefixed."""
    return {
        key.removeprefix("overall."): value
        for key, value in record.summary.items()
        if key.startswith("overall.")
    }


def _category_table(record: RunRecord) -> pd.DataFrame:
    """Build a category x metric pass-rate table from a run's summary."""
    categories = sorted({result.category for result in record.results})
    metrics = sorted({name for result in record.results for name in result.scores})
    table = pd.DataFrame(index=categories, columns=metrics, dtype=float)
    for category in categories:
        for metric in metrics:
            key = f"{category}.{metric}"
            if key in record.summary:
                table.loc[category, metric] = record.summary[key]
    return table


def _efficiency_table(record: RunRecord) -> pd.DataFrame:
    """Build a per-case efficiency table (optimal/actual/overage/retries)."""
    return pd.DataFrame(
        [
            {
                "case_id": result.case_id,
                "category": result.category,
                "optimal": result.optimal_tool_calls,
                "actual": result.actual_tool_calls,
                "overage": result.overage,
                "retries": result.retries,
            }
            for result in record.results
        ]
    ).set_index("case_id")


def _render_trace_drilldown(result: TestCaseResult) -> None:
    """Render one case's scores and every turn's full RunTrace."""
    st.write("Scores:", result.scores)
    for ref in result.trace_refs:
        path = Path(ref)
        if not path.exists():
            st.warning(f"Trace file not found: {ref}")
            continue
        trace = read_trace(path)
        with st.expander(f"Turn {trace.turn_id}: {trace.question}"):
            st.write("Resolved question:", trace.resolved_question)
            st.write("Route:", trace.route)
            st.write("Groundedness OK:", trace.groundedness_ok, " | Retries:", trace.retries)
            st.write("Ledger:", trace.ledger)
            st.write("Final answer:", trace.final_answer)


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Finance Q&A Eval Dashboard", layout="wide")
    st.title("Finance Q&A Agent — Eval Dashboard")

    runs = load_all_runs(DEFAULT_RUNS_DIR)
    if not runs:
        st.info("No eval runs found yet. Run `finance-qna eval run` first.")
        return

    latest = runs[-1]
    st.header(f"Latest run: {latest.run_id} ({latest.timestamp:%Y-%m-%d %H:%M})")
    st.caption(
        f"agent_id={latest.agent_id}  label={latest.label or '(none)'}  cases={len(latest.results)}"
    )

    st.subheader("Overall pass rate by metric")
    overall = _overall_metrics(latest)
    st.bar_chart(pd.Series(overall, name="pass rate"))

    st.subheader("Pass rate by category")
    st.dataframe(_category_table(latest).style.format("{:.0%}", na_rep="—"))

    st.subheader("Pass rate trend across runs")
    if overall:
        trend_metric = st.selectbox("Metric", sorted(overall.keys()))
        trend_rows = [
            {
                "run": f"{run.run_id} ({run.timestamp:%m-%d %H:%M})",
                "pass_rate": run.summary[f"overall.{trend_metric}"],
            }
            for run in runs
            if f"overall.{trend_metric}" in run.summary
        ]
        if trend_rows:
            st.line_chart(pd.DataFrame(trend_rows).set_index("run"))

    st.subheader("Efficiency (latest run)")
    st.dataframe(_efficiency_table(latest))

    st.subheader("Drill down into a case")
    case_id = st.selectbox("Case", [result.case_id for result in latest.results])
    selected = next(result for result in latest.results if result.case_id == case_id)
    _render_trace_drilldown(selected)


if __name__ == "__main__":
    main()
