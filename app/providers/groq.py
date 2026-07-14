from app.config import settings
from app.providers.openai_compat import OpenAICompatProvider


class GroqProvider(OpenAICompatProvider):
    """Provider 1: Groq's hosted API (OpenAI-compatible, free tier)."""

    name = "groq"

    @property
    def base_url(self) -> str:
        return settings.groq_base_url

    @property
    def api_key(self) -> str:
        return settings.groq_api_key

    @property
    def default_model(self) -> str:
        return settings.groq_default_model
