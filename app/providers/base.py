from abc import ABC, abstractmethod
from typing import Any, Dict

from app.models import ChatCompletionRequest


class ProviderError(Exception):
    """Raised when a provider call fails — the routing layer (Day 2)
    catches this to trigger fallback to the next provider."""

    def __init__(self, provider: str, message: str, status_code: int = 502):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class Provider(ABC):
    """Interface every provider (Groq, OpenRouter, llama.cpp) implements.

    The gateway only ever talks to this interface, so adding a provider
    means writing one new subclass — no changes to routing code.
    """

    name: str

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Execute a chat completion and return the OpenAI-format response dict.

        Must raise ProviderError on any failure (timeout, HTTP error, bad key)
        so the router can decide whether to fall back.
        """
        ...
