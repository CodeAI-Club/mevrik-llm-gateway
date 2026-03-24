"""
Per-user and per-model request statistics tracker.

Design for high concurrency:
- Writes go into an in-memory buffer (zero contention on hot path).
- A background flush loop persists to disk every N seconds.
- Reads serve from memory — no disk I/O on the request path.
- asyncio.Lock only held during flush and clear operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger("llm-gateway")

_FLUSH_INTERVAL = 30  # seconds


@dataclass
class RequestStat:
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    ttfb_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def tokens_per_sec(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return (self.tokens_out / self.latency_ms) * 1000


class StatsTracker:
    """High-throughput stats store with periodic file persistence."""

    def __init__(self, path: str, max_records_per_key: int = 1000):
        self._path = Path(path)
        self._max = max_records_per_key
        self._data: dict[str, list[RequestStat]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._dirty = False
        self._flush_task: Optional[asyncio.Task] = None
        self._load()

    # -- persistence --

    def _load(self):
        if not self._path.exists():
            logger.info("Stats file not found — starting fresh")
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for user_key, records in raw.items():
                self._data[user_key] = [RequestStat(**r) for r in records]
            total = sum(len(v) for v in self._data.values())
            logger.info("Loaded %d stat record(s) for %d user(s)", total, len(self._data))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Corrupt stats file, resetting: %s", exc)

    async def save(self):
        """Flush current state to disk."""
        async with self._lock:
            self._write()

    def _write(self):
        data = {
            user: [asdict(r) for r in records]
            for user, records in self._data.items()
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)
        self._dirty = False

    # -- background flush --

    def start_flush_loop(self):
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL)
            if self._dirty:
                async with self._lock:
                    self._write()
                    logger.debug("Stats flushed to disk")

    # -- write (lock-free hot path, lock only on trim) --

    async def record(self, user_key: str, stat: RequestStat):
        """Record a stat — fast append, no lock on the common path."""
        bucket = self._data[user_key]
        bucket.append(stat)
        self._dirty = True
        # Trim only when well over limit (amortised)
        if len(bucket) > self._max + 100:
            async with self._lock:
                self._data[user_key] = bucket[-self._max:]

    # -- read (no lock, serves from memory) --

    def _get(self, user_key: str, model_id: Optional[str] = None) -> list[RequestStat]:
        records = self._data.get(user_key, [])
        if model_id:
            records = [r for r in records if r.model_id == model_id]
        return records

    def summary(self, user_key: str, model_id: Optional[str] = None) -> dict:
        records = self._get(user_key, model_id)
        if not records:
            return {
                "total_requests": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "avg_latency_ms": 0,
                "avg_ttfb_ms": 0,
                "avg_tokens_per_sec": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
            }

        latencies = sorted(r.latency_ms for r in records)
        tps_values = [r.tokens_per_sec for r in records if r.tokens_per_sec > 0]
        n = len(records)

        def _pct(arr: list[float], p: float) -> float:
            idx = min(int(len(arr) * p), len(arr) - 1)
            return round(arr[idx], 2)

        return {
            "total_requests": n,
            "total_tokens_in": sum(r.tokens_in for r in records),
            "total_tokens_out": sum(r.tokens_out for r in records),
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "avg_ttfb_ms": round(sum(r.ttfb_ms for r in records) / n, 2),
            "avg_tokens_per_sec": round(sum(tps_values) / len(tps_values), 2) if tps_values else 0,
            "p50_latency_ms": _pct(latencies, 0.50),
            "p95_latency_ms": _pct(latencies, 0.95),
            "p99_latency_ms": _pct(latencies, 0.99),
        }

    def all_users(self) -> list[str]:
        return list(self._data.keys())

    async def clear(self, user_key: Optional[str] = None):
        async with self._lock:
            if user_key:
                self._data.pop(user_key, None)
            else:
                self._data.clear()
            self._write()


tracker = StatsTracker(settings.stats_path)
