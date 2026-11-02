"""HTTP helper with retries, timeouts and a small in-process TTL cache."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests

from config import settings
from utils.logging_conf import get_logger

log = get_logger("http")

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Any]] = {}
DEFAULT_HEADERS = {
    "User-Agent": "SureStatsBot/1.0 (+https://surestats.app)",
    "Accept": "application/json",
}


def cache_get(key: str, ttl: int | None = None) -> Any | None:
    ttl = settings.cache_ttl_seconds if ttl is None else ttl
    with _LOCK:
        item = _CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl:
        with _LOCK:
            _CACHE.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    with _LOCK:
        _CACHE[key] = (time.time(), value)


def cache_clear(prefix: str | None = None) -> int:
    with _LOCK:
        if prefix is None:
            n = len(_CACHE)
            _CACHE.clear()
            return n
        keys = [k for k in _CACHE if k.startswith(prefix)]
        for k in keys:
            _CACHE.pop(k, None)
        return len(keys)


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    cache_key: str | None = None,
    ttl: int | None = None,
    retries: int = 2,
) -> Any:
    """GET a JSON document. Returns None on failure (never raises)."""
    if cache_key:
        hit = cache_get(cache_key, ttl)
        if hit is not None:
            log.debug("cache hit %s", cache_key)
            return hit

    merged = dict(DEFAULT_HEADERS)
    if headers:
        merged.update(headers)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=merged, timeout=settings.http_timeout)
            if resp.status_code == 200:
                data = resp.json()
                if cache_key:
                    cache_set(cache_key, data)
                return data
            if resp.status_code in (401, 403):
                log.warning("auth rejected (%s) for %s", resp.status_code, url)
                return None
            if resp.status_code == 429:
                log.warning("rate limited on %s", url)
                time.sleep(1.5 * (attempt + 1))
                continue
            log.warning("HTTP %s for %s", resp.status_code, url)
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_err = exc
            log.warning("request failed (%s) attempt %s: %s", url, attempt + 1, exc)
            time.sleep(0.6 * (attempt + 1))

    if last_err:
        log.error("giving up on %s: %s", url, last_err)
    return None
