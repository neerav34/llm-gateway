import logging
import time
import uuid
from typing import List, Optional

import httpx

from app.config import settings

logger = logging.getLogger("gateway.ratelimit")


class SlidingWindowLimiter:
    """Per-key sliding window rate limiter on Upstash Redis.

    Each key owns a sorted set of request timestamps. On every request:
      1. ZREMRANGEBYSCORE  — evict timestamps older than the window
      2. ZADD              — record this request
      3. ZCARD             — count requests inside the window
      4. EXPIRE            — let idle keys clean themselves up

    All four run in ONE Upstash pipeline call (one HTTP round trip).
    Unlike a fixed window, the limit can't be doubled by bursting at a
    window boundary — the window slides with each request.

    Fails open: if Redis is unreachable, requests are allowed (consistent
    with the cache — availability over strict enforcement for this demo).
    """

    async def _pipeline(self, commands: List[List[str]]) -> Optional[list]:
        try:
            async with httpx.AsyncClient(timeout=settings.cache_timeout) as client:
                response = await client.post(
                    f"{settings.upstash_redis_rest_url}/pipeline",
                    headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
                    json=commands,
                )
        except httpx.HTTPError as exc:
            logger.warning("redis unavailable (%s): rate limit fails open", exc)
            return None
        if response.status_code != 200:
            logger.warning("redis error HTTP %s: %s", response.status_code, response.text[:200])
            return None
        return response.json()

    @property
    def enabled(self) -> bool:
        return bool(settings.upstash_redis_rest_url and settings.upstash_redis_rest_token)

    async def check(self, api_key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if not self.enabled:
            return True

        now = time.time()
        window = settings.rate_limit_window_seconds
        redis_key = f"ratelimit:{api_key}"
        member = f"{now}:{uuid.uuid4().hex[:8]}"  # unique so same-ms requests all count

        results = await self._pipeline([
            ["ZREMRANGEBYSCORE", redis_key, "0", str(now - window)],
            ["ZADD", redis_key, str(now), member],
            ["ZCARD", redis_key],
            ["EXPIRE", redis_key, str(window)],
        ])
        if results is None:
            return True  # fail open

        count = results[2].get("result", 0)
        if count > settings.rate_limit_requests:
            # Rejected requests shouldn't consume window slots
            await self._pipeline([["ZREM", redis_key, member]])
            return False
        return True
