"""Load-test the gateway's own overhead, not provider speed.

Two phases:
  1. /health          — network RTT + framework baseline (no gateway logic)
  2. cache-hit chat   — full gateway path (auth, rate limit check, Redis
                        cache lookup, response assembly) with ZERO provider
                        time: every request after the first is a cache HIT.

Client latency includes the internet between you and the server, so we also
collect the server-reported X-Gateway-Latency-Ms header — that is the
gateway's internal processing time (including its Redis round trips) and
the honest "overhead added by the gateway" number.

Requests rotate across API keys to stay inside the per-key rate limit.

Usage:
    python scripts/loadtest.py https://llm-gateway-rb0k.onrender.com \
        --requests 300 --concurrency 10
"""

import argparse
import asyncio
import statistics
import time

import httpx

PROMPT = "Load test prompt: what is a gateway? One sentence."


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, round(p / 100 * (len(values) - 1)))
    return values[idx]


def report(label, client_ms, server_ms=None):
    print(f"\n{label} ({len(client_ms)} requests)")
    print(f"  client latency  p50={pct(client_ms, 50):7.1f}ms  "
          f"p95={pct(client_ms, 95):7.1f}ms  p99={pct(client_ms, 99):7.1f}ms  "
          f"mean={statistics.mean(client_ms):7.1f}ms")
    if server_ms:
        print(f"  server-internal p50={pct(server_ms, 50):7.1f}ms  "
              f"p95={pct(server_ms, 95):7.1f}ms  p99={pct(server_ms, 99):7.1f}ms  "
              f"mean={statistics.mean(server_ms):7.1f}ms")


async def run_phase(base_url, path, n, concurrency, chat=False):
    client_ms, server_ms, statuses, cache = [], [], {}, {}
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def one(i):
            async with sem:
                started = time.perf_counter()
                try:
                    if chat:
                        r = await client.post(
                            f"{base_url}{path}",
                            headers={"X-API-Key": f"loadtest-{i % 50}"},
                            json={"messages": [{"role": "user", "content": PROMPT}],
                                  "max_tokens": 60},
                        )
                    else:
                        r = await client.get(f"{base_url}{path}")
                except httpx.HTTPError as exc:
                    statuses[f"error:{type(exc).__name__}"] = \
                        statuses.get(f"error:{type(exc).__name__}", 0) + 1
                    return
                elapsed = (time.perf_counter() - started) * 1000
                statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
                if r.status_code == 200:
                    client_ms.append(elapsed)
                    if "x-gateway-latency-ms" in r.headers:
                        server_ms.append(float(r.headers["x-gateway-latency-ms"]))
                    hit = r.headers.get("x-gateway-cache", "-")
                    cache[hit] = cache.get(hit, 0) + 1

        if chat:  # seed the cache so the measured phase is pure HITs
            await one(0)
            client_ms.clear(); server_ms.clear(); cache.clear(); statuses.clear()

        await asyncio.gather(*(one(i) for i in range(n)))

    return client_ms, server_ms, statuses, cache


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Target: {base}   requests={args.requests} concurrency={args.concurrency}")

    t0 = time.perf_counter()
    client_ms, _, statuses, _ = await run_phase(base, "/health", 100, args.concurrency)
    report("PHASE 1  /health baseline (network + framework)", client_ms)
    print(f"  statuses: {statuses}")

    client_ms, server_ms, statuses, cache = await run_phase(
        base, "/v1/chat/completions", args.requests, args.concurrency, chat=True
    )
    elapsed = time.perf_counter() - t0
    report("PHASE 2  chat cache-hit path (full gateway, no provider)", client_ms, server_ms)
    print(f"  statuses: {statuses}   cache: {cache}")
    print(f"  effective throughput: {len(client_ms) / elapsed:.1f} req/s "
          f"(concurrency {args.concurrency}, single free-tier instance)")


if __name__ == "__main__":
    asyncio.run(main())
