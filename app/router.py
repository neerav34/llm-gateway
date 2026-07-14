import logging
from typing import Any, AsyncIterator, Dict, List, Tuple

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

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> Tuple[str, AsyncIterator[bytes], List[str]]:
        """Streaming with pre-first-byte fallback.

        We pull the FIRST chunk here, inside the try — provider failures
        (bad key, connect error, non-200) all surface before any bytes are
        yielded, so we can still move to the next provider. Once that first
        chunk exists we're committed: bytes already sent to the client
        can't be unsent, so mid-stream failures end the stream instead of
        falling back (mixing two models' outputs would be worse).
        """
        errors: List[ProviderError] = []
        for provider in self.providers:
            stream = provider.chat_completion_stream(request)
            try:
                first = await stream.__anext__()
            except StopAsyncIteration:
                errors.append(ProviderError(provider.name, "empty stream"))
                continue
            except ProviderError as exc:
                logger.warning("provider %s failed: %s", provider.name, exc)
                errors.append(exc)
                continue

            if errors:
                logger.warning(
                    "streaming fell back to %s after: %s",
                    provider.name,
                    "; ".join(str(e) for e in errors),
                )

            async def replay() -> AsyncIterator[bytes]:
                yield first
                async for chunk in stream:
                    yield chunk

            return provider.name, replay(), [e.provider for e in errors]
        raise AllProvidersFailed(errors)
