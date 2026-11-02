"""
SQLite persistence layer for SureStats Bot.

Tables
------
analyses        : every value-bet analysis ever produced (audit trail)
odds_snapshots  : raw bookmaker quotes captured per run
backtests       : model validation metrics
kv_store        : generic key/value (calibration params, seed version...)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import settings
from utils.logging_conf import get_logger

log = get_logger("db")
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    sport             TEXT NOT NULL,
    match             TEXT NOT NULL,
    event_time        TEXT,
    selection         TEXT,
    market            TEXT,
    true_prob         REAL,
    book_prob         REAL,
    best_odds         REAL,
    best_book         REAL,
    value_pct         REAL,
    edge_pct          REAL,
    signal            TEXT,
    confidence        REAL,
    risk              TEXT,
    reasoning         TEXT,
    payload           TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_sport   ON analyses(sport, created_at DESC);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    sport        TEXT NOT NULL,
    match        TEXT NOT NULL,
    market       TEXT NOT NULL,
    bookmaker    TEXT NOT NULL,
    price        REAL NOT NULL,
    point        REAL
);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(match, created_at DESC);

CREATE TABLE IF NOT EXISTS backtests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    sport         TEXT NOT NULL,
    model_version TEXT,
    n_bets        INTEGER,
    accuracy      REAL,
    brier         REAL,
    log_loss      REAL,
    roi_pct       REAL,
    detail        TEXT
);

CREATE TABLE IF NOT EXISTS kv_store (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connection(db_path: Path | str | None = None):
    path = str(db_path or settings.db_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    with _LOCK, connection(db_path) as conn:
        conn.executescript(SCHEMA)
    log.info("database ready at %s", db_path or settings.db_path)


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def save_analysis(record: dict[str, Any]) -> int:
    cols = (
        "sport", "match", "event_time", "selection", "market", "true_prob",
        "book_prob", "best_odds", "best_book", "value_pct", "edge_pct",
        "signal", "confidence", "risk", "reasoning", "payload",
    )
    values = []
    for c in cols:
        v = record.get(c)
        if c == "payload" and v is not None and not isinstance(v, str):
            v = json.dumps(v, default=str)
        values.append(v)

    with _LOCK, connection() as conn:
        cur = conn.execute(
            f"INSERT INTO analyses (created_at, {', '.join(cols)}) "
            f"VALUES (?, {', '.join(['?'] * len(cols))})",
            [_now(), *values],
        )
        return int(cur.lastrowid)


def recent_analyses(limit: int = 50, sport: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM analyses"
    params: list[Any] = []
    if sport and sport.lower() != "all":
        q += " WHERE sport = ?"
        params.append(sport.lower())
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with connection() as conn:
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def analysis_stats() -> dict[str, Any]:
    with connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) n,
                      AVG(value_pct) avg_value,
                      SUM(CASE WHEN signal LIKE 'VALUE%' THEN 1 ELSE 0 END) greens,
                      AVG(confidence) avg_conf
               FROM analyses"""
        ).fetchone()
    d = dict(row) if row else {}
    return {k: (v if v is not None else 0) for k, v in d.items()}


def delete_all_analyses() -> None:
    with _LOCK, connection() as conn:
        conn.execute("DELETE FROM analyses")


# ---------------------------------------------------------------------------
# odds snapshots
# ---------------------------------------------------------------------------
def save_odds_snapshot(sport: str, match: str, market: str, quotes: Iterable[dict[str, Any]]) -> int:
    rows = [
        (_now(), sport, match, market, q.get("bookmaker", "?"), float(q.get("price", 0)),
         q.get("point"))
        for q in quotes
        if q.get("price")
    ]
    if not rows:
        return 0
    with _LOCK, connection() as conn:
        conn.executemany(
            "INSERT INTO odds_snapshots (created_at, sport, match, market, bookmaker, price, point)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def odds_history(match: str, market: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    q = "SELECT * FROM odds_snapshots WHERE match = ?"
    params: list[Any] = [match]
    if market:
        q += " AND market = ?"
        params.append(market)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with connection() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


# ---------------------------------------------------------------------------
# backtests
# ---------------------------------------------------------------------------
def save_backtest(result: dict[str, Any]) -> int:
    with _LOCK, connection() as conn:
        cur = conn.execute(
            """INSERT INTO backtests
               (created_at, sport, model_version, n_bets, accuracy, brier, log_loss, roi_pct, detail)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                _now(), result.get("sport"), result.get("model_version"),
                result.get("n_bets"), result.get("accuracy"), result.get("brier"),
                result.get("log_loss"), result.get("roi_pct"),
                json.dumps(result.get("detail", {}), default=str),
            ),
        )
        return int(cur.lastrowid)


def latest_backtest(sport: str | None = None) -> dict[str, Any] | None:
    q = "SELECT * FROM backtests"
    params: list[Any] = []
    if sport:
        q += " WHERE sport = ?"
        params.append(sport)
    q += " ORDER BY id DESC LIMIT 1"
    with connection() as conn:
        row = conn.execute(q, params).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["detail"] = json.loads(d.get("detail") or "{}")
    except json.JSONDecodeError:
        d["detail"] = {}
    return d


# ---------------------------------------------------------------------------
# kv store
# ---------------------------------------------------------------------------
def kv_set(key: str, value: Any) -> None:
    with _LOCK, connection() as conn:
        conn.execute(
            "INSERT INTO kv_store (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), _now()),
        )


def kv_get(key: str, default: Any = None) -> Any:
    with connection() as conn:
        row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return default
