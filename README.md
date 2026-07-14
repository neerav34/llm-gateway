# LLM Gateway

A multi-provider LLM routing gateway: one OpenAI-compatible endpoint in front of
multiple providers, with automatic fallback, response caching, per-key rate
limiting, cost tracking, and SSE streaming passthrough.

**Live:** https://llm-gateway-rb0k.onrender.com &nbsp;·&nbsp; **Demo video:** _coming soon_

> Free-tier hosting note: the instance sleeps after 15 min idle — the first
> request after a quiet spell takes ~30–50 s to cold-start. Everything after
> that is warm.

## Why

Every team shipping LLM features hits the same problems: providers go down or
rate-limit you, identical prompts get paid for twice, one leaked key can drain
a budget, and nobody knows which feature spent what. A gateway centralizes the
answers: clients make one API call, and routing, fallback, caching, limits, and
accounting happen in one place. This project is a working, deployed
implementation of that pattern — the same problem LiteLLM, OpenRouter, and
internal AI-infra teams solve.

## Try it

```bash
curl https://llm-gateway-rb0k.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-name-here" \
  -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}]}'
```

Run it twice — watch `x-gateway-cache: MISS` become `HIT`. Add `"stream": true`
for SSE. The endpoint is OpenAI-compatible, so any OpenAI SDK works by pointing
`base_url` at the gateway.

## Architecture

```
Client ──► FastAPI gateway (Render)
              │
              ├─ 1. auth: per-key identity (X-API-Key / Bearer)
              ├─ 2. rate limit: sliding window, sorted set in Redis ──► Upstash Redis
              ├─ 3. cache lookup: sha256(prompt+params) ────────────► Upstash Redis
              │      └─ on miss: semantic lookup (cosine ≥ 0.85) ───► Upstash Vector
              ├─ 4. route: Groq ──(on failure)──► OpenRouter
              │           (streaming: SSE passthrough, no buffering)
              └─ 5. usage log: tokens + est. $ per key ─────────────► SQLite
```

Providers implement one interface ([app/providers/base.py](app/providers/base.py));
both current providers are thin subclasses of a shared OpenAI-compatible base,
so adding a third (e.g. a local llama.cpp server — same protocol) is a
config-only subclass.

## Measured performance

All numbers self-measured against the deployed free-tier instance
(0.1 CPU, single instance) with [scripts/loadtest.py](scripts/loadtest.py) —
300 requests, concurrency 10, cache-hit path. Methodology: the cache-hit path
exercises everything the gateway *adds* (auth, rate limiter, Redis cache,
response assembly) with zero provider time, so it isolates gateway overhead
from provider speed.

| Metric (server-internal) | p50 | p95 | p99 |
|---|---|---|---|
| Cache-hit request | 473 ms | 520 ms | 925 ms |

Throughput: **10.3 req/s** sustained at concurrency 10 on the free instance,
300/300 requests served from cache, zero errors. The p50 is dominated by two
cross-region Redis round trips (Render ↔ Upstash); colocating them or merging
the rate-limit and cache reads into one pipelined call would cut it further.

**The story behind the numbers:** the first load test measured a 3.8 s p50 —
every Redis/provider call was opening a fresh HTTPS connection, and per-request
TLS handshakes crushed the 0.1-CPU instance. Switching to a single pooled
`httpx.AsyncClient` (app lifespan) took p50 from 3,802 ms to 473 ms and
throughput from 1.9 to 10.3 req/s. Commit `7ffb304` has the before/after.

## Design decisions

**Fallback — when does it trigger?** On timeout, network error, non-200 status,
or an error-shaped 200 body (OpenRouter does that). Providers are tried in
priority order; the response tells you what happened via `X-Gateway-Provider`
and `X-Gateway-Fallback-From`. Model names are provider-specific, so a fallback
provider uses its own default model rather than blindly forwarding the
requested one.

**Streaming + fallback.** The router pulls the *first* SSE chunk before
returning the stream to the client, so connection/auth/status failures still
fall back. After the first byte we're committed: sent bytes can't be unsent,
and splicing a second model's output mid-stream is worse than an honest error.

**Cache strategy — two layers.** Layer 1 is exact-match: key = sha256 of
canonical JSON (model, messages, temperature, max_tokens); any change is a
different key. On exact miss, layer 2 is semantic: the prompt is embedded by
Upstash Vector's hosted model and matched against previously cached prompts by
cosine similarity — a paraphrase ("when did humans first land on the Moon?" vs
"what year did people first set foot on the Moon?") serves the cached answer
as `SEMANTIC-HIT`, with the similarity score exposed in a response header.

Vectors store only a pointer to the Redis entry, so the response body has one
home and one TTL (1 h default); a semantic match whose Redis entry expired
deletes its stale vector and counts as a miss. The 0.85 threshold was
calibrated by measurement, not guessed: real paraphrases scored ~0.89–0.92,
unrelated prompts ~0.56 — the risk with semantic caching is returning a
"similar enough" answer that's actually wrong, so the threshold sits far above
the unrelated zone and is tunable per deployment (`SEMANTIC_CACHE_THRESHOLD`).
Streams bypass both layers.

**What if Redis dies?** The gateway fails open, verified by test: with Redis
unreachable, requests still serve (uncached, unlimited) rather than erroring.
Availability over strict enforcement — the right trade for a gateway whose
callers have their own provider keys. Redis calls have a 2 s budget so a
hanging Redis can't stall requests.

**Why sliding window over fixed window?** A fixed 20/min window lets a client
burst 40 requests in the two seconds around a window boundary (20 at 11:59:59,
20 more at 12:00:00). The sliding window (sorted set of timestamps, pruned on
each request) counts the *trailing* 60 s, so the limit holds at every instant.
Rejected requests get a 429 with `Retry-After` and don't consume window slots.
The evict+add+count+expire runs as one pipelined Upstash call.

**Cost tracking.** Every request logs provider, model, tokens, and estimated $
per API key to SQLite; `GET /v1/usage?api_key=...` returns month-to-date totals
and a per-provider breakdown. Cache hits are logged at $0 — the provider was
never called. For streams, providers append a final usage chunk
(`stream_options.include_usage`) which the gateway sniffs as it passes through.

**Scaling to multiple instances.** Rate limiting and caching already live in
shared Redis, so N gateway instances enforce one consistent limit. The two
single-instance pieces are SQLite usage logs (would move to Redis/Postgres)
and in-process state (there is none). Render free tier's disk is ephemeral, so
the usage DB resets on redeploy — fine for a demo, called out here for honesty.

## Run locally

```bash
git clone https://github.com/neerav34/llm-gateway && cd llm-gateway
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your keys (Groq, OpenRouter, Upstash)
.venv/bin/uvicorn app.main:app --reload
```

Only `GROQ_API_KEY` is strictly required to serve requests; without Upstash
credentials the gateway runs with caching and rate limiting disabled (fail-open
by design). Deploys to Render via [render.yaml](render.yaml) blueprint.

## Endpoints

| Route | What |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming supported |
| `GET /v1/usage?api_key=` | Month-to-date tokens + estimated cost per key |
| `GET /health` | Liveness |

Response headers on every completion: `X-Gateway-Provider`,
`X-Gateway-Cache` (HIT/SEMANTIC-HIT/MISS/BYPASS), `X-Gateway-Latency-Ms`,
`X-Gateway-Cache-Similarity` on semantic hits, and `X-Gateway-Fallback-From`
when a fallback occurred.

## Roadmap

- Local llama.cpp provider (config-only subclass; local dev-mode path)
- Merge rate-limit + cache reads into one Redis pipeline call
- Observability endpoint: per-provider request counts, cache hit rate, latency percentiles
