import logging
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.models import ChatCompletionRequest
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider
from app.router import AllProvidersFailed, Router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LLM Gateway", version="0.2.0")

# Priority order: Groq first (fastest), OpenRouter as fallback.
# llama.cpp joins as provider 3 in Day 5.
router = Router([GroqProvider(), OpenRouterProvider()])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    started = time.perf_counter()
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

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    headers = {
        "X-Gateway-Provider": served_by,
        "X-Gateway-Latency-Ms": str(elapsed_ms),
    }
    if failed:
        headers["X-Gateway-Fallback-From"] = ",".join(failed)
    return JSONResponse(content=result, headers=headers)
