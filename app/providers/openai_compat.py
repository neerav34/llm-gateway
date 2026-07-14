from typing import Any, AsyncIterator, Dict

import httpx

from app.config import settings
from app.models import ChatCompletionRequest
from app.providers.base import Provider, ProviderError


class OpenAICompatProvider(Provider):
    """Shared implementation for any OpenAI-compatible chat API
    (Groq, OpenRouter, llama.cpp server, ...). Subclasses supply
    connection details; this class owns request/stream mechanics."""

    name: str
    # Whether a client-supplied model name is meaningful for this provider.
    # False for fallback providers whose model namespace differs from the
    # primary's (e.g. Groq model names don't exist on OpenRouter).
    use_client_model = True

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    @property
    def api_key(self) -> str:
        raise NotImplementedError

    @property
    def default_model(self) -> str:
        raise NotImplementedError

    def _require_key(self) -> None:
        if not self.api_key:
            raise ProviderError(self.name, "API key is not configured", status_code=500)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _payload(self, request: ChatCompletionRequest, stream: bool) -> Dict[str, Any]:
        model = (request.model if self.use_client_model else None) or self.default_model
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload["stream"] = True
            # Ask for a final usage chunk so the gateway can track tokens/cost
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _check_body(self, data: Dict[str, Any]) -> None:
        """Hook: subclasses may reject error-shaped 200 responses."""

    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        self._require_key()
        try:
            async with httpx.AsyncClient(timeout=settings.provider_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(request, stream=False),
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
        self._check_body(data)
        return data

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[bytes]:
        """Yield raw SSE bytes. Raises ProviderError BEFORE the first yield
        if the connection or status fails, so the router can still fall back.
        After the first byte is yielded we're committed to this provider."""
        self._require_key()
        # No read timeout: tokens trickle in for as long as generation runs
        timeout = httpx.Timeout(settings.provider_timeout, read=None)
        client = httpx.AsyncClient(timeout=timeout)
        try:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(request, stream=True),
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise ProviderError(
                            self.name,
                            f"HTTP {response.status_code}: {body.decode(errors='replace')[:300]}",
                            status_code=502,
                        )
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.TimeoutException:
                raise ProviderError(self.name, "request timed out", status_code=504)
            except httpx.HTTPError as exc:
                raise ProviderError(self.name, f"network error: {exc}")
        finally:
            await client.aclose()
