"""Universal application configuration and the LLM provider factory.

`Settings` holds only what every architecture needs regardless of which one is
selected: credentials, which architecture to run, and where the data lives.
Anything specific to one architecture -- which model(s) it uses, tool-loop
limits, retry counts -- belongs to that architecture's own config instead (e.g.
`agent.langgraph_adapter.LangGraphAgentConfig`), per
design/eval-harness-plan.md §2: model/architecture choice must not leak into
shared, architecture-agnostic code.
"""

import logging
from functools import lru_cache
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The google-genai SDK logs a one-time warning recommending its own Chat
# session wrapper over calling Models.generate_content directly with tools
# bound ("automatic function calling"). That recommendation doesn't apply to
# us -- LangChain's ChatGoogleGenerativeAI.bind_tools() manages the tool loop
# itself, at a layer above this SDK -- so it's just noise on every call that
# binds tools. Raising this logger's level suppresses it without touching any
# other google-genai/langchain log output.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


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
