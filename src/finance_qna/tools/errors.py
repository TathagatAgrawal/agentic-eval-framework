"""Structured errors raised by the tool layer.

These are caught by the LangChain tool adapters (not raised through to the agent
as a Python exception) so the agent gets a clean, structured observation to reason
about instead of a stack trace.
"""


class CategoryNotFoundError(ValueError):
    """Raised when a tool is called with a category name that isn't in the dataset."""

    def __init__(self, category: str, valid_categories: list[str]) -> None:
        """Store the invalid category and the full list of valid ones for the caller."""
        self.category = category
        self.valid_categories = valid_categories
        super().__init__(f"no such category: '{category}'")
