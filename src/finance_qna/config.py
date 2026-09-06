"""Universal application configuration and the LLM provider factory.

`Settings` holds only what every architecture needs regardless of which one is
selected: credentials, which architecture to run, and where the data lives.
Anything specific to one architecture -- which model(s) it uses, tool-loop
limits, retry counts -- belongs to that architecture's own config instead (e.g.
`agent.langgraph_adapter.LangGraphAgentConfig`), per
design/eval-harness-plan.md §2: model/architecture choice must not leak into
shared, architecture-agnostic code.
"""

from functools import lru_cache
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / a `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: SecretStr
    agent_architecture: str = "langgraph"  # which AgentAdapter to build; see agent/adapter.py
    db_path: Path = Path("data/synthetic_transactions.db")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings` instance, loaded once and cached."""
    return Settings()  # type: ignore[call-arg]


def get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Build a `ChatGoogleGenerativeAI` client for `model_name` using the configured API key.

    `model_name` is a plain Gemini model id, not a `Settings` field -- callers
    (an architecture's own config, e.g. `LangGraphAgentConfig.agent_model`)
    decide what to pass here, so this stays reusable by any architecture.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
    )
