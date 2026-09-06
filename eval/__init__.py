"""The evaluation harness: a model- and architecture-agnostic test set, scorers,
runner, and regression store for the finance Q&A agent.

Deliberately kept outside `src/finance_qna` -- it's a repo-local evaluation tool
that drives an `AgentAdapter` (see `eval.adapters`), not a library the agent
itself depends on, the same way `tests/` isn't part of the shipped package.
"""
