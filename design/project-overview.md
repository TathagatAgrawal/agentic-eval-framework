# Personal Financial Statement Q&A Agent

## Goals

- Build an agentic system that answers natural-language questions about personal financial transactions (spending by category, trends, comparisons, anomalies) in a way that is accurate, transparent, and verifiable — across both single questions and multi-turn conversations.
- Use this project as a testbed for a rigorous evaluation harness, since financial Q&A has **deterministic ground truth** — every answer can be checked against the actual underlying data, unlike open-ended chat or RAG tasks where correctness is fuzzy.
- Make the Q&A agent first, and then add the evaluation harness later.

## Objectives

1. Demonstrate agentic tool-use design (an agent that reasons about a question, calls the right tools, and grounds its answer in real data) without unnecessary architectural complexity.
2. Produce a system where every numeric claim in an answer can be traced back to an actual query result.
3. Support natural follow-up questions that depend on conversational context, without losing groundedness or accuracy.
4. Build an evaluation harness that scores the agent on objective, rule-based criteria (numeric correctness, groundedness, refusal/clarification behavior, contextual correctness) rather than relying primarily on LLM-as-judge scoring.
5. Produce a concrete, quantified before/after result (e.g., "caught X% of ungrounded answers," "eval score improved from Y% to Z% after a prompt fix") that can anchor a resume bullet or interview story.

## Features

**Core agent capabilities:**

- Answer simple aggregation questions (e.g., "How much did I spend on food last month?")
- Answer multi-condition and comparative questions (e.g., "Did I spend more on dining or groceries last quarter?")
- Detect and describe trends or anomalies (e.g., "Did any subscription price change this year?")
- **Maintain conversational memory across turns**, so follow-up questions resolve correctly:
  - Implicit reference resolution (e.g., "and what about April?" after a question about March spending)
  - Scope carry-over (e.g., "what about just groceries?" narrowing a previous category question)
  - Comparative follow-ups (e.g., "is that more or less than last year?" referencing a prior answer)
- Ask a clarifying question when a query is ambiguous (e.g., "this month" is unclear, or a follow-up reference is genuinely unresolvable) instead of guessing
- Gracefully decline questions that are out of scope (e.g., unrelated to the financial data) instead of fabricating an answer
- Show its reasoning/query transparently alongside the final answer, so the user can verify how the number was derived

**Evaluation harness capabilities:**

- A labeled test set of questions spanning difficulty levels: simple lookups, multi-condition filters, comparisons, trend detection, ambiguous questions, out-of-scope questions, **and multi-turn conversation sequences**
- Automatic scoring of:
  - **Numeric correctness** — does the claimed number match the true value?
  - **Groundedness** — does every number in the answer trace back to a real query result (hallucination check)?
  - **Contextual correctness** — in multi-turn sequences, does the agent correctly resolve references to prior turns (right category, right time period, right comparison baseline)?
  - **Clarification handling** — does it ask when it should, instead of guessing (including when a follow-up reference is ambiguous)?
  - **Refusal correctness** — does it decline when a question is out of scope?
  - **Efficiency** — how many steps/tool calls did it take to answer?
- Regression tracking across runs, so changes to prompts or models can be evaluated against a consistent benchmark over time
- A simple dashboard view summarizing pass rates by category and trends across eval runs

## Non-Goals (for v1)

- No real personal/sensitive financial data — synthetic data only, to keep it portfolio-safe
- No cross-session memory (conversation history persists only within a single session, not saved/recalled across separate sessions)
- No production deployment or user-facing UI — CLI or minimal interface is sufficient
