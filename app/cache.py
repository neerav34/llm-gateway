import hashlib
import json
import logging
from typing import Any, Dict, Optional

import httpx

from app import http
from app.config import settings
from app.models import ChatCompletionRequest

logger = logging.getLogger("gateway.cache")


def cache_key(request: ChatCompletionRequest) -> str:
    """Exact-match key: hash of everything that affects the response.

    Canonical JSON (sorted keys, no whitespace) so logically identical
    requests always hash the same.
    """
    payload = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "cache:" + hashlib.sha256(canonical.encode()).hexdigest()


class RedisCache:
    """Exact-match response cache on Upstash Redis (REST API).

    Fails open: any Redis error/timeout logs a warning and behaves like a
    miss, so the gateway keeps serving when Redis is down — just uncached.
    """

    @property
    def enabled(self) -> bool:
        return bool(settings.upstash_redis_rest_url and settings.upstash_redis_rest_token)

    async def _command(self, *args: str) -> Optional[Any]:
        """Run one Redis command via Upstash REST (body = JSON array)."""
        try:
            response = await http.client().post(
                settings.upstash_redis_rest_url,
                headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
                json=list(args),
                timeout=settings.cache_timeout,
            )
        except httpx.HTTPError as exc:
            logger.warning("redis unavailable (%s): failing open", exc)
            return None
        if response.status_code != 200:
            logger.warning("redis error HTTP %s: %s", response.status_code, response.text[:200])
            return None
        return response.json().get("result")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        raw = await self._command("GET", key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        await self._command(
            "SET", key, json.dumps(value), "EX", str(settings.cache_ttl_seconds)
        )
