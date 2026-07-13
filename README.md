# LLM Gateway

A multi-provider LLM routing gateway with fallback, semantic caching, per-key rate
limiting, and cost tracking. Work in progress — full README (architecture diagram,
live URL, demo video, measured load-test numbers) lands at the end of the build.

## Status

- [x] Day 1 — Bare gateway, one provider (Groq)
- [x] Day 2 — Second provider (OpenRouter) + fallback routing
- [x] Day 3 — Exact-match caching on Upstash Redis (semantic caching: stretch goal)
- [ ] Day 4 — Per-key rate limiting + cost tracking
- [ ] Day 5 — Local provider (llama.cpp) + streaming passthrough
- [ ] Day 6 — Deploy to Render
- [ ] Day 7 — README, demo video, load test

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
uvicorn app.main:app --reload
```

## Try it

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}]}'
```

The endpoint is OpenAI-compatible, so any OpenAI SDK works by pointing its
`base_url` at the gateway.
