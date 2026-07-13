from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway settings, loaded from environment variables / .env file."""

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_default_model: str = "llama-3.1-8b-instant"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "nvidia/nemotron-nano-9b-v2:free"

    # Seconds to wait for a provider before giving up (fallback trigger in Day 2)
    provider_timeout: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
