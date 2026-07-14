import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import http
from app.cache import RedisCache, cache_key
from app.config import settings
from app.models import ChatCompletionRequest
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider
from app.ratelimit import SlidingWindowLimiter
from app.router import AllProvidersFailed, Router
from app.usage import UsageTracker

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await http.startup()
    yield
    await http.shutdown()


app = FastAPI(title="LLM Gateway", version="0.5.0", lifespan=lifespan)

# Priority order: Groq first (fastest), OpenRouter as fallback.
# llama.cpp joins as provider 3 in Day 5.
router = Router([GroqProvider(), OpenRouterProvider()])
cache = RedisCache()
limiter = SlidingWindowLimiter()
usage_tracker = UsageTracker()


def client_api_key(request: Request) -> str:
    """Identify the caller: X-API-Key header, or Authorization: Bearer."""
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return "anonymous"


def month_start_ts() -> float:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/usage")
async def usage(api_key: Optional[str] = None, this_month: bool = True):
    """Token + estimated cost report, optionally for one key."""
    since = month_start_ts() if this_month else 0.0
    return usage_tracker.summary(api_key=api_key, since_ts=since)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    started = time.perf_counter()
    api_key = client_api_key(http_request)

    # Rate limit before doing any work — cached responses count too
    if not await limiter.check(api_key):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": (
                        f"rate limit exceeded: {settings.rate_limit_requests} requests"
                        f" per {settings.rate_limit_window_seconds}s"
                    )
                }
            },
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )

    if request.stream:
        return await _stream_completion(request, api_key, started)

    use_cache = cache.enabled
    key = cache_key(request) if use_cache else None

    if use_cache:
        cached = await cache.get(key)
        if cached is not None:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            response = cached["response"]
            usage_tracker.log(
                api_key=api_key,
                provider=cached.get("provider", "unknown"),
                model=response.get("model"),
                usage=response.get("usage") or {},
                latency_ms=elapsed_ms,
                cache_hit=True,
            )
            return JSONResponse(
                content=response,
                headers={
                    "X-Gateway-Cache": "HIT",
                    "X-Gateway-Provider": cached.get("provider", "unknown"),
                    "X-Gateway-Latency-Ms": str(elapsed_ms),
                },
            )

    try:
        served_by, result, failed = await router.chat_completion(request)
    except AllProvidersFailed as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "all providers failed",
                    "details": [str(e) for e in exc.errors],
                }
            },
        )

    if use_cache:
        await cache.set(key, {"provider": served_by, "response": result})

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    usage_tracker.log(
        api_key=api_key,
        provider=served_by,
        model=result.get("model"),
        usage=result.get("usage") or {},
        latency_ms=elapsed_ms,
        cache_hit=False,
    )

    headers = {
        "X-Gateway-Cache": "MISS" if use_cache else "BYPASS",
        "X-Gateway-Provider": served_by,
        "X-Gateway-Latency-Ms": str(elapsed_ms),
    }
    if failed:
        headers["X-Gateway-Fallback-From"] = ",".join(failed)
    return JSONResponse(content=result, headers=headers)


async def _stream_completion(request: ChatCompletionRequest, api_key: str, started: float):
    """SSE passthrough: provider chunks stream to the client verbatim.

    Streams bypass the cache (a token stream isn't a cacheable JSON blob).
    Usage is still tracked: providers append a final usage chunk
    (stream_options.include_usage), which we sniff as it passes through.
    """
    try:
        served_by, stream, failed = await router.chat_completion_stream(request)
    except AllProvidersFailed as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "all providers failed",
                    "details": [str(e) for e in exc.errors],
                }
            },
        )

    async def passthrough():
        sniffed = {"model": None, "usage": None}
        buffer = b""
        async for chunk in stream:
            yield chunk  # client latency first — sniffing happens after send
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                sniffed["model"] = obj.get("model") or sniffed["model"]
                if obj.get("usage"):
                    sniffed["usage"] = obj["usage"]
        usage_tracker.log(
            api_key=api_key,
            provider=served_by,
            model=sniffed["model"],
            usage=sniffed["usage"] or {},
            latency_ms=round((time.perf_counter() - started) * 1000),
            cache_hit=False,
        )

    headers = {
        "X-Gateway-Cache": "BYPASS",
        "X-Gateway-Provider": served_by,
    }
    if failed:
        headers["X-Gateway-Fallback-From"] = ",".join(failed)
    return StreamingResponse(passthrough(), media_type="text/event-stream", headers=headers)
