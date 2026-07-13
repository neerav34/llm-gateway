import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.models import ChatCompletionRequest
from app.providers.base import ProviderError
from app.providers.groq import GroqProvider

app = FastAPI(title="LLM Gateway", version="0.1.0")

# Day 1: single hardcoded provider. Day 2 replaces this with a router
# that tries providers in order and falls back on failure.
provider = GroqProvider()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    started = time.perf_counter()
    try:
        result = await provider.chat_completion(request)
    except ProviderError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc), "provider": exc.provider}},
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return JSONResponse(
        content=result,
        headers={
            "X-Gateway-Provider": provider.name,
            "X-Gateway-Latency-Ms": str(elapsed_ms),
        },
    )
