import logging
from typing import Any, Dict, Optional, Tuple

import httpx

from app import http
from app.config import settings
from app.models import ChatCompletionRequest

logger = logging.getLogger("gateway.semantic_cache")


class SemanticCache:
    """Second cache layer: paraphrase-tolerant lookup via Upstash Vector.

    Upstash Vector hosts the embedding model, so we upsert/query raw text.
    Vectors carry only a POINTER (the exact-match Redis key) — the response
    body lives in Redis and keeps its TTL. A match whose Redis entry has
    expired deletes its stale vector and counts as a miss.

    Known trade-off: similarity is judged on the prompt text only, not on
    model/temperature/max_tokens. A paraphrase asked with different params
    can hit a response generated with the original's params. Acceptable
    here; a metadata filter on params is the next refinement.

    Fails open, like the exact cache and the rate limiter.
    """

    @property
    def enabled(self) -> bool:
        return bool(settings.upstash_vector_rest_url and settings.upstash_vector_rest_token)

    async def _post(self, path: str, body: Any) -> Optional[Any]:
        try:
            response = await http.client().post(
                f"{settings.upstash_vector_rest_url}{path}",
                headers={"Authorization": f"Bearer {settings.upstash_vector_rest_token}"},
                json=body,
                timeout=settings.cache_timeout,
            )
        except httpx.HTTPError as exc:
            logger.warning("vector store unavailable (%s): failing open", exc)
            return None
        if response.status_code != 200:
            logger.warning(
                "vector store HTTP %s: %s", response.status_code, response.text[:200]
            )
            return None
        return response.json().get("result")

    @staticmethod
    def _text(request: ChatCompletionRequest) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in request.messages)

    async def lookup(self, request: ChatCompletionRequest) -> Optional[Tuple[str, float]]:
        """Return (redis_key_of_cached_response, similarity) or None."""
        result = await self._post(
            "/query-data",
            {"data": self._text(request), "topK": 1, "includeMetadata": True},
        )
        if not result:
            return None
        match = result[0]
        score = float(match.get("score", 0.0))
        redis_key = (match.get("metadata") or {}).get("redis_key")
        if not redis_key or score < settings.semantic_cache_threshold:
            return None
        return redis_key, score

    async def store(self, request: ChatCompletionRequest, redis_key: str) -> None:
        # id = the exact-match key: identical prompts overwrite in place
        await self._post(
            "/upsert-data",
            {
                "id": redis_key,
                "data": self._text(request),
                "metadata": {"redis_key": redis_key},
            },
        )

    async def evict(self, vector_id: str) -> None:
        """Drop a vector whose Redis entry has expired."""
        await self._post("/delete", [vector_id])
