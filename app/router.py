import logging
from typing import Any, Dict, List, Tuple

from app.models import ChatCompletionRequest
from app.providers.base import Provider, ProviderError

logger = logging.getLogger("gateway.router")


class AllProvidersFailed(Exception):
    def __init__(self, errors: List[ProviderError]):
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors))


class Router:
    """Tries providers in priority order; falls back on any ProviderError.

    Returns (serving_provider_name, response, failed_provider_names) so the
    API layer can report whether a fallback happened.
    """

    def __init__(self, providers: List[Provider]):
        self.providers = providers

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        errors: List[ProviderError] = []
        for provider in self.providers:
            try:
                result = await provider.chat_completion(request)
                if errors:
                    logger.warning(
                        "fell back to %s after: %s",
                        provider.name,
                        "; ".join(str(e) for e in errors),
                    )
                return provider.name, result, [e.provider for e in errors]
            except ProviderError as exc:
                logger.warning("provider %s failed: %s", provider.name, exc)
                errors.append(exc)
        raise AllProvidersFailed(errors)
