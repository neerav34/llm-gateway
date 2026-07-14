from typing import Any, Dict

from app.config import settings
from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider


class OpenRouterProvider(OpenAICompatProvider):
    """Provider 2: OpenRouter free models (OpenAI-compatible)."""

    name = "openrouter"
    # Model names are provider-specific (Groq's "llama-3.1-8b-instant"
    # doesn't exist here), so on fallback we always use our own default.
    use_client_model = False

    @property
    def base_url(self) -> str:
        return settings.openrouter_base_url

    @property
    def api_key(self) -> str:
        return settings.openrouter_api_key

    @property
    def default_model(self) -> str:
        return settings.openrouter_default_model

    def _check_body(self, data: Dict[str, Any]) -> None:
        # OpenRouter can return 200 with an error body (e.g. upstream 404)
        if "error" in data:
            raise ProviderError(self.name, f"provider error: {str(data['error'])[:300]}")
