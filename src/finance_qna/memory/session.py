"""In-process, single-session conversational memory.

Holds the structured `TurnMemory` history for one chat session. Deliberately not
persisted anywhere, per the project's non-goal of cross-session memory -- it
lives only as long as the process (or the CLI's `chat` loop) that created it.
"""

from finance_qna.agent.state import TurnMemory


class SessionMemory:
    """An ordered, in-memory list of completed turns for one chat session."""

    def __init__(self) -> None:
        """Start a new, empty session with no prior turns."""
        self.turns: list[TurnMemory] = []

    def append(self, turn: TurnMemory) -> None:
        """Record a newly completed turn."""
        self.turns.append(turn)

    def recent(self, n: int = 2) -> list[TurnMemory]:
        """Return the last `n` turns, oldest first (empty if there are none yet)."""
        return self.turns[-n:] if n > 0 else []
