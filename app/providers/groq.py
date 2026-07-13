from typing import Any, Dict

import httpx

from app.config import settings
from app.models import ChatCompletionRequest
from app.providers.base import Provider, ProviderError


class GroqProvider(Provider):
    """Provider 1: Groq's hosted API (OpenAI-compatible, free tier)."""

    name = "groq"

    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        if not settings.groq_api_key:
            raise ProviderError(self.name, "GROQ_API_KEY is not set", status_code=500)

        payload: Dict[str, Any] = {
            "model": request.model or settings.groq_default_model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            async with httpx.AsyncClient(timeout=settings.provider_timeout) as client:
                response = await client.post(
                    f"{settings.groq_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
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

        return response.json()
