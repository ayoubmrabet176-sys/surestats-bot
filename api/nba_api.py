"""
balldontlie.io client for NBA team context.

Pulls the team's recent games to derive last-10 form, rest days and
back-to-back status. Ratings (off/def) come from the bootstrap snapshot and
are nudged by observed scoring when live data is available. Falls back
cleanly when no key is configured.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import settings
from data.datasets import find_nba_team
from utils.http import get_json
from utils.logging_conf import get_logger

log = get_logger("nba_api")


def _headers() -> dict[str, str]:
    return {"Authorization": settings.balldontlie_api_key} if settings.balldontlie_api_key else {}


def find_team(name: str) -> dict[str, Any] | None:
    if not settings.has_nba_key:
        return None
    data = get_json(
        f"{settings.nba_base_url}/teams",
        headers=_headers(),
        cache_key="nba:teams", ttl=86400,
    )
    if not data:
        return None
    q = name.lower()
    for t in data.get("data", []):
        full = f"{t.get('city','')} {t.get('name','')}".lower()
        if q in full or full in q or q == (t.get("abbreviation", "").lower()):
            return t
    return None


def recent_games(team_id: int, limit: int = 10) -> list[dict[str, Any]]:
    if not settings.has_nba_key:
        return []
    season = datetime.now(timezone.utc).year
    data = get_json(
        f"{settings.nba_base_url}/games",
        params={"team_ids[]": team_id, "seasons[]": season, "per_page": 25},
        headers=_headers(),
        cache_key=f"nba:games:{team_id}:{season}",
        ttl=1800,
    )
    if not data:
        return []
    games = [g for g in data.get("data", []) if g.get("status") == "Final"]
    games.sort(key=lambda g: g.get("date", ""), reverse=True)
    return games[:limit]


def team_profile(name: str) -> dict[str, Any]:
    base = find_nba_team(name)
    if base is None:
        base = {
            "name": name.title(), "abbr": name[:3].upper(), "conf": "?", "elo": 1650,
            "pace": 99.5, "off_rtg": 112.0, "def_rtg": 114.0, "last10": 5,
            "home": (20, 20), "away": (15, 25), "rest_days": 2, "b2b": False,
            "injuries": [],
        }

    if not settings.has_nba_key:
        base["source"] = "bootstrap"
        return base

    team = find_team(name)
    if not team:
        base["source"] = "bootstrap (live lookup empty)"
        return base

    games = recent_games(team["id"])
    if not games:
        base["source"] = "bootstrap (no games returned)"
        return base

    wins = 0
    for g in games:
        home_id = g.get("home_team", {}).get("id")
        hs = g.get("home_team_score", 0)
        vs = g.get("visitor_team_score", 0)
        won = (hs > vs) if home_id == team["id"] else (vs > hs)
        wins += int(won)

    base["last10"] = wins
    last_date = games[0].get("date", "")
    try:
        d = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
        delta = (datetime.now(timezone.utc) - d).days
        base["rest_days"] = max(0, min(7, delta - 1))
        base["b2b"] = base["rest_days"] == 0
    except (ValueError, TypeError):
        pass

    try:
        scored = sum(g.get("home_team_score", 0) if g.get("home_team", {}).get("id") == team["id"]
                     else g.get("visitor_team_score", 0) for g in games) / len(games)
        allowed = sum(g.get("visitor_team_score", 0) if g.get("home_team", {}).get("id") == team["id"]
                      else g.get("home_team_score", 0) for g in games) / len(games)
        base["off_rtg"] = round(0.7 * base["off_rtg"] + 0.3 * scored, 1)
        base["def_rtg"] = round(0.7 * base["def_rtg"] + 0.3 * allowed, 1)
    except (TypeError, ZeroDivisionError):
        pass

    base["source"] = "balldontlie"
    return base


def injury_report(name: str) -> list[str]:
    """
    balldontlie's free tier does not expose an injury feed, so we keep the
    curated report and tag the source. Paid plans can plug their endpoint in
    here - the shape is a list of strings.
    """
    prof = find_nba_team(name)
    return list(prof.get("injuries", [])) if prof else []
