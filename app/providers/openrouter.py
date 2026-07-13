from typing import Any, Dict

import httpx

from app.config import settings
from app.models import ChatCompletionRequest
from app.providers.base import Provider, ProviderError


class OpenRouterProvider(Provider):
    """Provider 2: OpenRouter free models (OpenAI-compatible)."""

    name = "openrouter"

    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        if not settings.openrouter_api_key:
            raise ProviderError(self.name, "OPENROUTER_API_KEY is not set", status_code=500)

        payload: Dict[str, Any] = {
            # A client-supplied model name is provider-specific (e.g. Groq's
            # "llama-3.1-8b-instant" doesn't exist on OpenRouter), so on
            # fallback we always use this provider's own default model.
            "model": settings.openrouter_default_model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            async with httpx.AsyncClient(timeout=settings.provider_timeout) as client:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                    json=payload,
                )
        except httpx.TimeoutException:
            raise ProviderError(self.name, "request timed out", status_code=504)
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc}")

        if response.status_code != 200:
            raise ProviderError(
                self.name,
                f"HTTP {response.status_code}: {response.text[:300]}",
                status_code=502,
            )

        data = response.json()
        # OpenRouter can return 200 with an error body (e.g. upstream 404)
        if "error" in data:
            raise ProviderError(self.name, f"provider error: {str(data['error'])[:300]}")
        return data
