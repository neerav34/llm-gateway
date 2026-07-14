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

    # Upstash Redis (REST API) — cache + rate limit store
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    cache_ttl_seconds: int = 3600
    # Budget for a cache lookup; a dead Redis must not stall requests
    cache_timeout: float = 2.0

    # Upstash Vector (REST) — semantic cache. Unset = exact-match caching only.
    upstash_vector_rest_url: str = ""
    upstash_vector_rest_token: str = ""
    # Min cosine similarity to serve a semantically-cached response.
    # Calibrated against the index's embedding model: true paraphrases
    # score ~0.89+, unrelated prompts ~0.56 — 0.85 sits far above the
    # false-positive zone while catching real rewordings.
    semantic_cache_threshold: float = 0.85

    # Per-API-key rate limit: max requests per sliding window
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # SQLite file for per-key token/cost tracking
    usage_db_path: str = "usage.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
