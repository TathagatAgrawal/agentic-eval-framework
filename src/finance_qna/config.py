"""Application configuration and the LLM provider factory.

Centralizing `ChatGoogleGenerativeAI` construction here means model choice is
swappable per agent node without touching node logic, and is also what the
(future) evaluation harness uses to run the same test set against a different
model or prompt version for regression comparison.
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
    agent_model: str = "gemini-3.5-flash-lite"
    classifier_model: str = "gemini-3.5-flash-lite"
    db_path: Path = Path("data/synthetic_transactions.db")
    max_tool_steps: int = 6
    groundedness_retry_limit: int = 1


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings` instance, loaded once and cached."""
    return Settings()  # type: ignore[call-arg]


def get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Build a `ChatGoogleGenerativeAI` client for `model_name` using the configured API key.

    `model_name` is expected to be one of the model names in `Settings`
    (e.g. `settings.agent_model` or `settings.classifier_model`), not necessarily
    a literal Gemini model id, so callers stay decoupled from the specific model.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
    )
