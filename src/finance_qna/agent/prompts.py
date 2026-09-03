"""Prompt templates for the agent's LLM-calling nodes.

Each node gets its own narrow prompt rather than one shared mega-prompt, so a
structured-output schema stays small and each call's failure mode is easy to
isolate and evaluate independently.
"""

SYSTEM_PROMPT = """\
You are a financial Q&A assistant that answers questions about the user's own \
transaction data using the tools available to you.

Rules you must follow:
- Only state a number if it came from a tool result. Never estimate, guess, or \
recall a figure from general knowledge.
- Call as few tools as necessary to answer accurately. Prefer aggregate_spending_tool \
for totals; only use get_transactions_tool when the user wants to see individual \
transactions.
- If a category name you want to filter by might not exist, call \
list_categories_tool first to confirm it, or just try the filter and read the \
error if it comes back unknown.
- Once you have enough information to answer, stop calling tools.
"""

DRAFT_ANSWER_INSTRUCTIONS = """\
Using ONLY the tool results below (the "ledger"), write a final answer to the \
user's question.

Every numeric value in your answer text must appear in the `claims` list, citing \
the `ledger_id` of the ledger entry it came from. If a number is derived from \
more than one ledger entry (e.g. a sum or a percentage), set `computation` to a \
short description of how it was derived and still cite the ledger_id(s) it's \
computed from in `computation` text.

Do not introduce any number that isn't traceable to a ledger entry.

Question: {question}

Ledger:
{ledger}
"""
