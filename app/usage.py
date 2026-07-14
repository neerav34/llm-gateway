import sqlite3
import time
from typing import Any, Dict, Optional

from app.config import settings

# USD per 1M tokens: (input, output). Unknown models (e.g. :free ones)
# track tokens but cost $0. Update rates from provider pricing pages.
PRICING = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    api_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    est_cost_usd REAL DEFAULT 0,
    latency_ms INTEGER,
    cache_hit INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_log(api_key, ts);
"""


def estimate_cost(model: Optional[str], prompt_tokens: int, completion_tokens: int) -> float:
    input_rate, output_rate = PRICING.get(model or "", (0.0, 0.0))
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


class UsageTracker:
    """Per-key token/cost log in SQLite.

    Plain synchronous sqlite3: writes are single-row inserts (<1ms), fine
    for a single-process gateway. Multi-instance deployments would move
    this to a shared store (same Redis, or Postgres).
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(settings.usage_db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def log(
        self,
        api_key: str,
        provider: str,
        model: Optional[str],
        usage: Dict[str, Any],
        latency_ms: int,
        cache_hit: bool,
    ) -> None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        # A cache hit costs nothing — the provider was never called
        cost = 0.0 if cache_hit else estimate_cost(model, prompt_tokens, completion_tokens)
        self._conn.execute(
            "INSERT INTO usage_log (ts, api_key, provider, model, prompt_tokens,"
            " completion_tokens, est_cost_usd, latency_ms, cache_hit)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                api_key,
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                cost,
                latency_ms,
                int(cache_hit),
            ),
        )
        self._conn.commit()

    def summary(self, api_key: Optional[str] = None, since_ts: float = 0.0) -> Dict[str, Any]:
        where = "ts >= ?"
        params: list = [since_ts]
        if api_key:
            where += " AND api_key = ?"
            params.append(api_key)
        rows = self._conn.execute(
            f"""SELECT api_key, provider,
                       COUNT(*) as requests,
                       SUM(prompt_tokens) as prompt_tokens,
                       SUM(completion_tokens) as completion_tokens,
                       ROUND(SUM(est_cost_usd), 6) as est_cost_usd,
                       SUM(cache_hit) as cache_hits
                FROM usage_log WHERE {where}
                GROUP BY api_key, provider ORDER BY api_key, provider""",
            params,
        ).fetchall()
        totals = self._conn.execute(
            f"""SELECT COUNT(*), SUM(prompt_tokens + completion_tokens),
                       ROUND(SUM(est_cost_usd), 6), SUM(cache_hit)
                FROM usage_log WHERE {where}""",
            params,
        ).fetchone()
        return {
            "totals": {
                "requests": totals[0] or 0,
                "total_tokens": totals[1] or 0,
                "est_cost_usd": totals[2] or 0.0,
                "cache_hits": totals[3] or 0,
            },
            "breakdown": [
                {
                    "api_key": r[0],
                    "provider": r[1],
                    "requests": r[2],
                    "prompt_tokens": r[3] or 0,
                    "completion_tokens": r[4] or 0,
                    "est_cost_usd": r[5] or 0.0,
                    "cache_hits": r[6] or 0,
                }
                for r in rows
            ],
        }
