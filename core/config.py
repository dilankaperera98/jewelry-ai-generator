import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str


def load_settings() -> Settings:
    """Load application settings from the environment and an optional .env file."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not configured")
    return Settings(anthropic_api_key=api_key)
