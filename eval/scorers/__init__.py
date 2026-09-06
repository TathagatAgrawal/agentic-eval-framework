"""The six deterministic scorers, one per metric in design/eval-harness-plan.md §3.

Every scorer is a pure function of `(RunTrace, ExpectedResult)` -- no LLM calls,
no I/O -- so the whole scoring layer is testable with hand-built fixtures and
never spends API quota. None of them import LangGraph or any agent-internal
type; only `RunTrace` (from `finance_qna.tracing.trace`) and, for groundedness,
the already-framework-independent `finance_qna.agent.groundedness.verify`.
"""
