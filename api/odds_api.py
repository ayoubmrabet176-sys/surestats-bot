"""
The Odds API integration (the-odds-api.com) with a deterministic fallback.

Live path
---------
Uses /v4/sports to discover active competitions, then /v4/sports/{key}/odds
with h2h + totals markets and a wide bookmaker list. Every quote is kept so
we can compute the average (market) probability and the single best price.

Fallback path
-------------
If no ODDS_API_KEY is configured (or the API is unreachable / out of
credits), a deterministic market simulator produces a full 24-bookmaker
board. The simulated market is deliberately imperfect - it leans on a
public-style estimate with per-book noise and margins - which is exactly
what creates realistic value opportunities. It is seeded by the match name
plus a 10-minute time bucket, so the board is stable within a refresh cycle
and moves between cycles, just like a real market.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any

from config import settings
from utils.http import get_json
from utils.logging_conf import get_logger

log = get_logger("odds")

BOOKMAKERS = [
    "Bet365", "Pinnacle", "William Hill", "Betfair", "1xBet", "Unibet",
    "DraftKings", "FanDuel", "BetMGM", "Caesars", "PointsBet", "Betway",
    "Bwin", "Marathonbet", "888sport", "Betsson", "NordicBet", "Coolbet",
    "SBOBET", "Matchbook", "Smarkets", "Betano", "Stake", "Cloudbet",
]

# typical per-book margin (overround) - Pinnacle/Matchbook are sharp, retail books are not
BOOK_MARGIN = {
    "Pinnacle": 0.022, "Matchbook": 0.020, "Smarkets": 0.021, "Betfair": 0.028,
    "1xBet": 0.035, "SBOBET": 0.038, "Marathonbet": 0.040, "Bet365": 0.045,
    "William Hill": 0.048, "DraftKings": 0.052, "FanDuel": 0.050, "BetMGM": 0.051,
    "Caesars": 0.053, "PointsBet": 0.055, "Betway": 0.056, "Bwin": 0.057,
    "888sport": 0.058, "Betsson": 0.056, "NordicBet": 0.057, "Coolbet": 0.049,
    "Unibet": 0.054, "Betano": 0.052, "Stake": 0.044, "Cloudbet": 0.040,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bucket(seconds: int = 600) -> int:
    return int(datetime.now(timezone.utc).timestamp() // seconds)


def _seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


def _det_rand(seed: int, index: int) -> float:
    """Deterministic pseudo-random in [0,1) from a seed and an index."""
    x = math.sin((seed % 100000) * 0.0001 + index * 12.9898) * 43758.5453
    return x - math.floor(x)


def _decimal(price: Any) -> float | None:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    return p if p > 1.0 else None


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------
def active_sports() -> list[dict[str, Any]]:
    if not settings.has_odds_key:
        return []
    data = get_json(
        f"{settings.odds_base_url}/sports/",
        params={"apiKey": settings.odds_api_key},
        cache_key="odds:sports", ttl=3600,
    )
    return data if isinstance(data, list) else []


def _sport_keys(sport: str) -> list[str]:
    if sport == "nba":
        return ["basketball_nba"]
    sports = active_sports()
    keys = [s.get("key", "") for s in sports if s.get("active") and "tennis" in s.get("key", "")]
    priority = ["atp", "wta", "grand", "slam", "aus_open", "wimbledon", "us_open", "french"]
    keys.sort(key=lambda k: next((i for i, p in enumerate(priority) if p in k), 99))
    return keys[:4]


def _match_event(events: list[dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    def norm(s: str) -> str:
        return s.lower().replace(".", "").replace("-", " ").strip()

    na, nb = norm(a), norm(b)
    best = None
    for ev in events:
        h, v = norm(ev.get("home_team", "")), norm(ev.get("away_team", ""))
        if (na in h or h in na) and (nb in v or v in nb):
            return ev
        if (na in v or v in na) and (nb in h or h in nb):
            return ev
        if na.split()[-1] in h + v and nb.split()[-1] in h + v:
            best = best or ev
    return best


def fetch_live_odds(sport: str, a: str, b: str) -> dict[str, Any] | None:
    if not settings.has_odds_key:
        return None

    for key in _sport_keys(sport):
        data = get_json(
            f"{settings.odds_base_url}/sports/{key}/odds/",
            params={
                "apiKey": settings.odds_api_key,
                "regions": "eu,us,uk,au",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
                "bookmakers": ",".join(b.lower().replace(" ", "") for b in BOOKMAKERS),
            },
            cache_key=f"odds:{key}:{a}:{b}", ttl=settings.odds_refresh_seconds,
        )
        if not isinstance(data, list) or not data:
            continue
        ev = _match_event(data, a, b)
        if not ev:
            continue

        moneyline, totals = [], []
        line = None
        for bk in ev.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for market in bk.get("markets", []):
                if market.get("key") == "h2h":
                    for oc in market.get("outcomes", []):
                        price = _decimal(oc.get("price"))
                        if price:
                            moneyline.append({"bookmaker": book, "selection": oc.get("name"), "price": price})
                elif market.get("key") == "totals":
                    for oc in market.get("outcomes", []):
                        price = _decimal(oc.get("price"))
                        if price:
                            point = oc.get("point")
                            line = point if line is None else line
                            totals.append({"bookmaker": book, "selection": oc.get("name"),
                                           "price": price, "point": point})
        if moneyline:
            log.info("live odds: %s vs %s via %s (%s books)", a, b, key,
                     len({q['bookmaker'] for q in moneyline}))
            return {
                "source": "live", "fetched_at": _now_iso(),
                "book_count": len({q["bookmaker"] for q in moneyline}),
                "moneyline": moneyline, "totals": totals, "line": line,
                "event": {"home": ev.get("home_team"), "away": ev.get("away_team"),
                          "commence": ev.get("commence_time"), "sport_key": key},
                "note": f"Live board from The Odds API ({key})",
            }
    log.info("no live odds found for %s vs %s - using simulated market", a, b)
    return None


# ---------------------------------------------------------------------------
# deterministic simulated market
# ---------------------------------------------------------------------------
def simulate_market(
    match: str,
    p_a: float,
    p_b: float,
    *,
    totals_line: float | None = None,
    p_over: float | None = None,
    market_bias: float = 0.0,
) -> dict[str, Any]:
    seed = _seed(match.lower(), _bucket(settings.odds_refresh_seconds))

    # the market's own read can drift a few points from the model - that drift
    # is what a value bettor is trying to exploit
    bias = market_bias + (_det_rand(seed, 1) - 0.5) * 0.085
    m_a = min(max(p_a + bias, 0.03), 0.97)

    moneyline = []
    for i, book in enumerate(BOOKMAKERS):
        margin = BOOK_MARGIN.get(book, 0.05)
        noise = (_det_rand(seed, 10 + i) - 0.5) * 0.045
        book_a = min(max(m_a + noise, 0.02), 0.98)
        book_b = 1 - book_a
        moneyline.append({"bookmaker": book, "selection": "A",
                          "price": round(1.0 / (book_a * (1 + margin / 2)), 3)})
        moneyline.append({"bookmaker": book, "selection": "B",
                          "price": round(1.0 / (book_b * (1 + margin / 2)), 3)})

    totals = []
    if totals_line is not None and p_over is not None:
        m_over = min(max(p_over + (_det_rand(seed, 2) - 0.5) * 0.045, 0.05), 0.95)
        for i, book in enumerate(BOOKMAKERS):
            margin = BOOK_MARGIN.get(book, 0.05)
            noise = (_det_rand(seed, 40 + i) - 0.5) * 0.03
            bo = min(max(m_over + noise, 0.03), 0.97)
            bu = 1 - bo
            totals.append({"bookmaker": book, "selection": "Over",
                           "price": round(1.0 / (bo * (1 + margin / 2)), 3), "point": totals_line})
            totals.append({"bookmaker": book, "selection": "Under",
                           "price": round(1.0 / (bu * (1 + margin / 2)), 3), "point": totals_line})

    return {
        "source": "simulated", "fetched_at": _now_iso(), "book_count": len(BOOKMAKERS),
        "moneyline": moneyline, "totals": totals, "line": totals_line,
        "event": {"home": None, "away": None, "commence": None, "sport_key": "simulated"},
        "note": "Simulated 24-bookmaker board (no ODDS_API_KEY configured or no live event found)",
    }


def get_market(
    sport: str,
    match: str,
    a: str,
    b: str,
    p_a: float,
    p_b: float,
    *,
    totals_line: float | None = None,
    p_over: float | None = None,
) -> dict[str, Any]:
    """Live board if possible, otherwise the deterministic simulator."""
    live = fetch_live_odds(sport, a, b) if settings.live_data else None
    if live:
        return live
    return simulate_market(match, p_a, p_b, totals_line=totals_line, p_over=p_over)


# ---------------------------------------------------------------------------
# market maths
# ---------------------------------------------------------------------------
def best_price(quotes: list[dict[str, Any]], selection: str) -> tuple[float, str]:
    rows = [q for q in quotes if q.get("selection") == selection]
    if not rows:
        return 0.0, "-"
    top = max(rows, key=lambda q: q["price"])
    return float(top["price"]), str(top["bookmaker"])


def average_price(quotes: list[dict[str, Any]], selection: str) -> float:
    rows = [q["price"] for q in quotes if q.get("selection") == selection]
    return float(sum(rows) / len(rows)) if rows else 0.0


def devigged_probability(quotes: list[dict[str, Any]], sel_a: str, sel_b: str) -> float:
    """Average implied probability for sel_a after removing each book's vig."""
    probs = []
    by_book: dict[str, dict[str, float]] = {}
    for q in quotes:
        if q.get("selection") in (sel_a, sel_b) and q.get("price", 0) > 1:
            by_book.setdefault(q["bookmaker"], {})[q["selection"]] = 1.0 / q["price"]
    for _book, legs in by_book.items():
        if sel_a in legs and sel_b in legs:
            total = legs[sel_a] + legs[sel_b]
            if total > 0:
                probs.append(legs[sel_a] / total)
        elif sel_a in legs:
            probs.append(min(legs[sel_a], 0.99))
    return float(sum(probs) / len(probs)) if probs else 0.0


def market_spread(quotes: list[dict[str, Any]], selection: str) -> dict[str, float]:
    rows = [q["price"] for q in quotes if q.get("selection") == selection]
    if not rows:
        return {"min": 0, "max": 0, "avg": 0}
    return {"min": min(rows), "max": max(rows), "avg": sum(rows) / len(rows)}
