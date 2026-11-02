"""
API-Tennis (api-sports.io) client.

Live calls refresh ELO-relevant facts: recent results, serve stats,
head-to-head and injury notes. Everything degrades gracefully to the curated
bootstrap profile when a key is missing or the plan quota is spent - the
analysis never crashes because of a data provider.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import settings
from data.datasets import find_tennis_player
from utils.http import get_json
from utils.logging_conf import get_logger

log = get_logger("tennis_api")


def _headers() -> dict[str, str]:
    return {"x-apisports-key": settings.tennis_api_key}


def search_player(name: str) -> dict[str, Any] | None:
    if not settings.has_tennis_key:
        return None
    data = get_json(
        f"{settings.tennis_base_url}/players",
        params={"search": name},
        headers=_headers(),
        cache_key=f"tennis:player:{name.lower()}",
        ttl=86400,
    )
    if not data:
        return None
    rows = data.get("response") or []
    return rows[0] if rows else None


def player_profile(name: str) -> dict[str, Any]:
    """
    Returns a normalised profile. Live data (when available) is layered on top
    of the bootstrap profile so every downstream model has the same shape.
    """
    base = find_tennis_player(name)
    if base is None:
        base = {
            "name": name.title(), "tour": "ATP", "country": "?", "rank": 50,
            "elo": 1700, "elo_hard": 1700, "elo_clay": 1700, "elo_grass": 1700,
            "form": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "serve_pct": 0.60, "bp_saved": 0.60, "matches_7d": 2,
            "injury": None, "h2h": {},
        }

    if not settings.has_tennis_key:
        base["source"] = "bootstrap"
        return base

    raw = search_player(name)
    if not raw:
        base["source"] = "bootstrap (live lookup empty)"
        return base

    player = raw.get("player", raw) if isinstance(raw, dict) else {}
    stats = raw.get("statistics", []) if isinstance(raw, dict) else []
    try:
        if player.get("name"):
            base["name"] = player["name"]
        if isinstance(player.get("country"), dict) and player["country"].get("name"):
            base["country"] = player["country"]["name"]
        if isinstance(stats, list) and stats:
            s = stats[0]
            games = s.get("games", {}) or {}
            serve = games.get("serve", {}) or {}
            if serve.get("percentage"):
                base["serve_pct"] = float(str(serve["percentage"]).replace("%", "")) / 100
            bp = games.get("breakPoints", {}) or {}
            if bp.get("saved"):
                try:
                    saved = str(bp["saved"]).split("/")
                    base["bp_saved"] = float(saved[0]) / max(1.0, float(saved[1]))
                except (ValueError, IndexError):
                    pass
        base["source"] = "api-tennis"
    except (TypeError, ValueError, KeyError) as exc:
        log.warning("could not parse API-Tennis profile for %s: %s", name, exc)
        base["source"] = "bootstrap (parse error)"
    return base


def recent_form(name: str, n: int = 10) -> list[int]:
    """1 = win, most recent first."""
    if not settings.has_tennis_key:
        prof = find_tennis_player(name)
        return list(prof["form"]) if prof else []
    season = datetime.now(timezone.utc).year
    data = get_json(
        f"{settings.tennis_base_url}/games",
        params={"player": name, "season": season},
        headers=_headers(),
        cache_key=f"tennis:games:{name.lower()}:{season}",
        ttl=3600,
    )
    if not data:
        prof = find_tennis_player(name)
        return list(prof["form"]) if prof else []
    games = data.get("response") or []
    form: list[int] = []
    for g in sorted(games, key=lambda x: x.get("date", ""), reverse=True)[:n]:
        winner = (g.get("winner") or {}).get("name", "")
        form.append(1 if name.lower() in winner.lower() else 0)
    if form:
        return form
    prof = find_tennis_player(name)
    return list(prof["form"]) if prof else []


def head_to_head(a: str, b: str) -> list[int] | None:
    if not settings.has_tennis_key:
        return None
    data = get_json(
        f"{settings.tennis_base_url}/headtohead",
        params={"player1": a, "player2": b},
        headers=_headers(),
        cache_key=f"tennis:h2h:{a.lower()}:{b.lower()}",
        ttl=86400,
    )
    if not data:
        return None
    rows = data.get("response") or []
    if not rows:
        return None
    wins_a = wins_b = 0
    for g in rows:
        winner = (g.get("winner") or {}).get("name", "").lower()
        if a.lower().split()[-1] in winner:
            wins_a += 1
        elif b.lower().split()[-1] in winner:
            wins_b += 1
    return [wins_a, wins_b] if wins_a + wins_b else None
