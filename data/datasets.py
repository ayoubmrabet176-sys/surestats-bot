"""
Bootstrap datasets for SureStats Bot.

Two things live here:

1. REFERENCE PROFILES - a curated snapshot of ATP/WTA players and NBA teams
   (ELO, surface ELO, recent form, H2H, serve/break-point stats, pace,
   offensive/defensive rating, rest days, injuries). These are used as the
   fallback when a live API key is missing, and as the base prior that live
   API data updates on top of.

2. CALIBRATION HISTORY - a deterministic, seeded set of historical matches
   generated from those profiles. It is used to fit the scikit-learn
   logistic calibrator that maps our feature blend -> win probability, and
   to produce the historical-accuracy / ROI backtest. It is clearly labelled
   as bootstrap data; when live results are available they are appended and
   the calibrator is refit.

Nothing here is random at runtime: every generator is seeded, so the same
numbers appear on every machine.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TENNIS
# ---------------------------------------------------------------------------
# elo = overall ELO, surface elo per surface.
# form = last 10 results, most recent first (1 = win).
# h2h = {opponent: [wins_for, losses_for]}
TENNIS_PLAYERS: dict[str, dict[str, Any]] = {
    "novak djokovic": {
        "name": "Novak Djokovic", "tour": "ATP", "country": "SRB", "rank": 3,
        "elo": 2075, "elo_hard": 2105, "elo_clay": 2040, "elo_grass": 2110,
        "form": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        "serve_pct": 0.645, "bp_saved": 0.672, "matches_7d": 3,
        "injury": None,
        "h2h": {"carlos alcaraz": [5, 3], "jannik sinner": [4, 2], "daniil medvedev": [10, 5],
                "alexander zverev": [8, 4], "stefanos tsitsipas": [12, 2]},
    },
    "carlos alcaraz": {
        "name": "Carlos Alcaraz", "tour": "ATP", "country": "ESP", "rank": 2,
        "elo": 2085, "elo_hard": 2050, "elo_clay": 2155, "elo_grass": 2080,
        "form": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        "serve_pct": 0.632, "bp_saved": 0.641, "matches_7d": 4,
        "injury": "forearm - minor, playing with tape",
        "h2h": {"novak djokovic": [3, 5], "jannik sinner": [7, 3], "daniil medvedev": [6, 2],
                "alexander zverev": [5, 3]},
    },
    "jannik sinner": {
        "name": "Jannik Sinner", "tour": "ATP", "country": "ITA", "rank": 1,
        "elo": 2110, "elo_hard": 2160, "elo_clay": 2035, "elo_grass": 2060,
        "form": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        "serve_pct": 0.658, "bp_saved": 0.689, "matches_7d": 4,
        "injury": None,
        "h2h": {"novak djokovic": [2, 4], "carlos alcaraz": [3, 7], "daniil medvedev": [6, 3],
                "alexander zverev": [7, 2]},
    },
    "daniil medvedev": {
        "name": "Daniil Medvedev", "tour": "ATP", "country": "RUS", "rank": 5,
        "elo": 1985, "elo_hard": 2035, "elo_clay": 1870, "elo_grass": 1930,
        "form": [0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
        "serve_pct": 0.612, "bp_saved": 0.655, "matches_7d": 3,
        "injury": None,
        "h2h": {"novak djokovic": [5, 10], "carlos alcaraz": [2, 6], "jannik sinner": [3, 6]},
    },
    "alexander zverev": {
        "name": "Alexander Zverev", "tour": "ATP", "country": "GER", "rank": 4,
        "elo": 2010, "elo_hard": 2015, "elo_clay": 2045, "elo_grass": 1935,
        "form": [1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        "serve_pct": 0.671, "bp_saved": 0.701, "matches_7d": 3,
        "injury": None,
        "h2h": {"novak djokovic": [4, 8], "carlos alcaraz": [3, 5], "jannik sinner": [2, 7]},
    },
    "stefanos tsitsipas": {
        "name": "Stefanos Tsitsipas", "tour": "ATP", "country": "GRE", "rank": 11,
        "elo": 1915, "elo_hard": 1870, "elo_clay": 1990, "elo_grass": 1830,
        "form": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        "serve_pct": 0.618, "bp_saved": 0.628, "matches_7d": 2,
        "injury": "back - managed",
        "h2h": {"novak djokovic": [2, 12], "carlos alcaraz": [1, 4]},
    },
    "holger rune": {
        "name": "Holger Rune", "tour": "ATP", "country": "DEN", "rank": 13,
        "elo": 1900, "elo_hard": 1885, "elo_clay": 1935, "elo_grass": 1850,
        "form": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        "serve_pct": 0.605, "bp_saved": 0.617, "matches_7d": 3,
        "injury": None,
        "h2h": {"carlos alcaraz": [1, 3], "jannik sinner": [1, 4]},
    },
    "taylor fritz": {
        "name": "Taylor Fritz", "tour": "ATP", "country": "USA", "rank": 7,
        "elo": 1935, "elo_hard": 1965, "elo_clay": 1830, "elo_grass": 1900,
        "form": [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        "serve_pct": 0.663, "bp_saved": 0.664, "matches_7d": 4,
        "injury": None,
        "h2h": {"novak djokovic": [0, 4], "alexander zverev": [3, 4]},
    },
    "casper ruud": {
        "name": "Casper Ruud", "tour": "ATP", "country": "NOR", "rank": 8,
        "elo": 1910, "elo_hard": 1855, "elo_clay": 2015, "elo_grass": 1800,
        "form": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "serve_pct": 0.628, "bp_saved": 0.640, "matches_7d": 3,
        "injury": None,
        "h2h": {"carlos alcaraz": [1, 4], "novak djokovic": [0, 3]},
    },
    "andrey rublev": {
        "name": "Andrey Rublev", "tour": "ATP", "country": "RUS", "rank": 9,
        "elo": 1900, "elo_hard": 1925, "elo_clay": 1870, "elo_grass": 1840,
        "form": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "serve_pct": 0.617, "bp_saved": 0.601, "matches_7d": 4,
        "injury": None,
        "h2h": {"daniil medvedev": [2, 5], "jannik sinner": [1, 5]},
    },
    "alex de minaur": {
        "name": "Alex de Minaur", "tour": "ATP", "country": "AUS", "rank": 10,
        "elo": 1885, "elo_hard": 1910, "elo_clay": 1825, "elo_grass": 1875,
        "form": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        "serve_pct": 0.641, "bp_saved": 0.652, "matches_7d": 5,
        "injury": None,
        "h2h": {"novak djokovic": [1, 3], "taylor fritz": [4, 3]},
    },
    "ben shelton": {
        "name": "Ben Shelton", "tour": "ATP", "country": "USA", "rank": 15,
        "elo": 1855, "elo_hard": 1885, "elo_clay": 1770, "elo_grass": 1840,
        "form": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "serve_pct": 0.668, "bp_saved": 0.633, "matches_7d": 3,
        "injury": None,
        "h2h": {"taylor fritz": [1, 2]},
    },
    "iga swiatek": {
        "name": "Iga Swiatek", "tour": "WTA", "country": "POL", "rank": 1,
        "elo": 2080, "elo_hard": 2010, "elo_clay": 2190, "elo_grass": 1900,
        "form": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        "serve_pct": 0.655, "bp_saved": 0.681, "matches_7d": 3,
        "injury": None,
        "h2h": {"aryna sabalenka": [8, 4], "coco gauff": [11, 2], "elena rybakina": [5, 3]},
    },
    "aryna sabalenka": {
        "name": "Aryna Sabalenka", "tour": "WTA", "country": "BLR", "rank": 2,
        "elo": 2050, "elo_hard": 2095, "elo_clay": 1965, "elo_grass": 2010,
        "form": [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        "serve_pct": 0.661, "bp_saved": 0.648, "matches_7d": 4,
        "injury": None,
        "h2h": {"iga swiatek": [4, 8], "coco gauff": [5, 4], "elena rybakina": [6, 3]},
    },
    "coco gauff": {
        "name": "Coco Gauff", "tour": "WTA", "country": "USA", "rank": 3,
        "elo": 1990, "elo_hard": 2005, "elo_clay": 1950, "elo_grass": 1930,
        "form": [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        "serve_pct": 0.598, "bp_saved": 0.612, "matches_7d": 4,
        "injury": "shoulder - monitored",
        "h2h": {"iga swiatek": [2, 11], "aryna sabalenka": [4, 5]},
    },
    "elena rybakina": {
        "name": "Elena Rybakina", "tour": "WTA", "country": "KAZ", "rank": 5,
        "elo": 1965, "elo_hard": 1990, "elo_clay": 1880, "elo_grass": 2010,
        "form": [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
        "serve_pct": 0.672, "bp_saved": 0.671, "matches_7d": 3,
        "injury": None,
        "h2h": {"iga swiatek": [3, 5], "aryna sabalenka": [3, 6]},
    },
    "jessica pegula": {
        "name": "Jessica Pegula", "tour": "WTA", "country": "USA", "rank": 6,
        "elo": 1900, "elo_hard": 1925, "elo_clay": 1830, "elo_grass": 1860,
        "form": [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        "serve_pct": 0.622, "bp_saved": 0.639, "matches_7d": 3,
        "injury": None,
        "h2h": {"coco gauff": [3, 4], "iga swiatek": [1, 6]},
    },
    "ons jabeur": {
        "name": "Ons Jabeur", "tour": "WTA", "country": "TUN", "rank": 12,
        "elo": 1855, "elo_hard": 1830, "elo_clay": 1885, "elo_grass": 1900,
        "form": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "serve_pct": 0.601, "bp_saved": 0.608, "matches_7d": 2,
        "injury": "knee - doubtful",
        "h2h": {"iga swiatek": [2, 5], "aryna sabalenka": [2, 4]},
    },
}

# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------
NBA_TEAMS: dict[str, dict[str, Any]] = {
    "boston celtics": {"name": "Boston Celtics", "abbr": "BOS", "conf": "East", "elo": 1810,
                       "pace": 99.2, "off_rtg": 121.4, "def_rtg": 110.2, "last10": 8,
                       "home": (28, 8), "away": (23, 13), "rest_days": 2, "b2b": False, "injuries": []},
    "denver nuggets": {"name": "Denver Nuggets", "abbr": "DEN", "conf": "West", "elo": 1785,
                       "pace": 97.8, "off_rtg": 118.9, "def_rtg": 112.4, "last10": 7,
                       "home": (30, 7), "away": (20, 16), "rest_days": 1, "b2b": False, "injuries": []},
    "oklahoma city thunder": {"name": "Oklahoma City Thunder", "abbr": "OKC", "conf": "West", "elo": 1800,
                              "pace": 100.4, "off_rtg": 119.6, "def_rtg": 108.9, "last10": 8,
                              "home": (31, 6), "away": (24, 12), "rest_days": 2, "b2b": False, "injuries": []},
    "milwaukee bucks": {"name": "Milwaukee Bucks", "abbr": "MIL", "conf": "East", "elo": 1740,
                        "pace": 99.9, "off_rtg": 117.2, "def_rtg": 114.6, "last10": 6,
                        "home": (27, 9), "away": (18, 19), "rest_days": 1, "b2b": True,
                        "injuries": ["G. Antetokounmpo - questionable (calf)"]},
    "phoenix suns": {"name": "Phoenix Suns", "abbr": "PHX", "conf": "West", "elo": 1715,
                     "pace": 100.8, "off_rtg": 116.4, "def_rtg": 115.9, "last10": 5,
                     "home": (24, 12), "away": (16, 21), "rest_days": 0, "b2b": True,
                     "injuries": ["B. Beal - out (hamstring)"]},
    "los angeles lakers": {"name": "Los Angeles Lakers", "abbr": "LAL", "conf": "West", "elo": 1730,
                           "pace": 101.1, "off_rtg": 115.8, "def_rtg": 113.7, "last10": 6,
                           "home": (26, 11), "away": (17, 20), "rest_days": 2, "b2b": False,
                           "injuries": ["J. Vanderbilt - out (foot)"]},
    "golden state warriors": {"name": "Golden State Warriors", "abbr": "GSW", "conf": "West", "elo": 1725,
                              "pace": 101.6, "off_rtg": 116.9, "def_rtg": 114.8, "last10": 6,
                              "home": (25, 12), "away": (18, 19), "rest_days": 1, "b2b": False, "injuries": []},
    "dallas mavericks": {"name": "Dallas Mavericks", "abbr": "DAL", "conf": "West", "elo": 1750,
                         "pace": 99.4, "off_rtg": 118.1, "def_rtg": 113.1, "last10": 7,
                         "home": (26, 11), "away": (19, 18), "rest_days": 3, "b2b": False,
                         "injuries": ["D. Lively II - out (ankle)"]},
    "new york knicks": {"name": "New York Knicks", "abbr": "NYK", "conf": "East", "elo": 1760,
                        "pace": 97.2, "off_rtg": 117.6, "def_rtg": 111.8, "last10": 7,
                        "home": (28, 9), "away": (20, 16), "rest_days": 2, "b2b": False,
                        "injuries": ["M. Robinson - out (foot)"]},
    "minnesota timberwolves": {"name": "Minnesota Timberwolves", "abbr": "MIN", "conf": "West", "elo": 1745,
                               "pace": 98.6, "off_rtg": 114.9, "def_rtg": 109.6, "last10": 6,
                               "home": (27, 10), "away": (19, 18), "rest_days": 1, "b2b": False, "injuries": []},
    "cleveland cavaliers": {"name": "Cleveland Cavaliers", "abbr": "CLE", "conf": "East", "elo": 1770,
                            "pace": 98.1, "off_rtg": 119.1, "def_rtg": 110.9, "last10": 8,
                            "home": (30, 7), "away": (22, 14), "rest_days": 2, "b2b": False, "injuries": []},
    "philadelphia 76ers": {"name": "Philadelphia 76ers", "abbr": "PHI", "conf": "East", "elo": 1690,
                           "pace": 99.6, "off_rtg": 113.2, "def_rtg": 114.9, "last10": 4,
                           "home": (21, 15), "away": (14, 24), "rest_days": 1, "b2b": True,
                           "injuries": ["J. Embiid - out (knee)"]},
    "la clippers": {"name": "LA Clippers", "abbr": "LAC", "conf": "West", "elo": 1720,
                    "pace": 97.5, "off_rtg": 115.4, "def_rtg": 113.2, "last10": 6,
                    "home": (25, 12), "away": (18, 19), "rest_days": 2, "b2b": False,
                    "injuries": ["K. Leonard - questionable (knee)"]},
    "indiana pacers": {"name": "Indiana Pacers", "abbr": "IND", "conf": "East", "elo": 1700,
                       "pace": 103.4, "off_rtg": 118.4, "def_rtg": 117.1, "last10": 6,
                       "home": (24, 13), "away": (17, 20), "rest_days": 1, "b2b": False, "injuries": []},
    "miami heat": {"name": "Miami Heat", "abbr": "MIA", "conf": "East", "elo": 1705,
                   "pace": 96.9, "off_rtg": 112.8, "def_rtg": 111.4, "last10": 5,
                   "home": (24, 13), "away": (16, 21), "rest_days": 2, "b2b": False,
                   "injuries": ["T. Herro - day-to-day (ankle)"]},
    "orlando magic": {"name": "Orlando Magic", "abbr": "ORL", "conf": "East", "elo": 1710,
                      "pace": 97.1, "off_rtg": 112.4, "def_rtg": 109.8, "last10": 6,
                      "home": (26, 11), "away": (17, 20), "rest_days": 1, "b2b": False, "injuries": []},
    "sacramento kings": {"name": "Sacramento Kings", "abbr": "SAC", "conf": "West", "elo": 1695,
                         "pace": 100.2, "off_rtg": 115.1, "def_rtg": 115.4, "last10": 5,
                         "home": (23, 14), "away": (16, 21), "rest_days": 2, "b2b": False, "injuries": []},
    "houston rockets": {"name": "Houston Rockets", "abbr": "HOU", "conf": "West", "elo": 1735,
                        "pace": 98.9, "off_rtg": 114.6, "def_rtg": 109.9, "last10": 7,
                        "home": (27, 10), "away": (20, 16), "rest_days": 1, "b2b": False, "injuries": []},
    "memphis grizzlies": {"name": "Memphis Grizzlies", "abbr": "MEM", "conf": "West", "elo": 1720,
                          "pace": 102.6, "off_rtg": 116.2, "def_rtg": 113.4, "last10": 6,
                          "home": (25, 12), "away": (18, 19), "rest_days": 2, "b2b": False,
                          "injuries": ["J. Morant - questionable (shoulder)"]},
    "atlanta hawks": {"name": "Atlanta Hawks", "abbr": "ATL", "conf": "East", "elo": 1665,
                      "pace": 101.8, "off_rtg": 114.4, "def_rtg": 118.2, "last10": 4,
                      "home": (21, 15), "away": (14, 24), "rest_days": 1, "b2b": True, "injuries": []},
    "chicago bulls": {"name": "Chicago Bulls", "abbr": "CHI", "conf": "East", "elo": 1640,
                      "pace": 100.6, "off_rtg": 111.6, "def_rtg": 116.8, "last10": 3,
                      "home": (18, 18), "away": (12, 26), "rest_days": 2, "b2b": False, "injuries": []},
    "brooklyn nets": {"name": "Brooklyn Nets", "abbr": "BKN", "conf": "East", "elo": 1615,
                      "pace": 98.4, "off_rtg": 109.8, "def_rtg": 115.6, "last10": 3,
                      "home": (16, 20), "away": (11, 27), "rest_days": 1, "b2b": False, "injuries": []},
    "toronto raptors": {"name": "Toronto Raptors", "abbr": "TOR", "conf": "East", "elo": 1620,
                        "pace": 100.1, "off_rtg": 110.4, "def_rtg": 116.2, "last10": 3,
                        "home": (17, 19), "away": (10, 28), "rest_days": 2, "b2b": False, "injuries": []},
    "portland trail blazers": {"name": "Portland Trail Blazers", "abbr": "POR", "conf": "West", "elo": 1600,
                               "pace": 99.8, "off_rtg": 108.9, "def_rtg": 117.4, "last10": 2,
                               "home": (15, 21), "away": (9, 29), "rest_days": 1, "b2b": True, "injuries": []},
    "utah jazz": {"name": "Utah Jazz", "abbr": "UTA", "conf": "West", "elo": 1590,
                  "pace": 99.1, "off_rtg": 109.2, "def_rtg": 118.6, "last10": 2,
                  "home": (14, 22), "away": (8, 30), "rest_days": 2, "b2b": False, "injuries": []},
    "san antonio spurs": {"name": "San Antonio Spurs", "abbr": "SAS", "conf": "West", "elo": 1650,
                          "pace": 100.9, "off_rtg": 112.1, "def_rtg": 115.1, "last10": 4,
                          "home": (19, 17), "away": (12, 26), "rest_days": 1, "b2b": False, "injuries": []},
    "new orleans pelicans": {"name": "New Orleans Pelicans", "abbr": "NOP", "conf": "West", "elo": 1610,
                             "pace": 99.3, "off_rtg": 110.1, "def_rtg": 117.9, "last10": 2,
                             "home": (15, 21), "away": (9, 29), "rest_days": 2, "b2b": False,
                             "injuries": ["Z. Williamson - out (hamstring)"]},
    "detroit pistons": {"name": "Detroit Pistons", "abbr": "DET", "conf": "East", "elo": 1635,
                        "pace": 100.3, "off_rtg": 111.2, "def_rtg": 115.8, "last10": 4,
                        "home": (18, 18), "away": (12, 26), "rest_days": 1, "b2b": False, "injuries": []},
    "charlotte hornets": {"name": "Charlotte Hornets", "abbr": "CHA", "conf": "East", "elo": 1585,
                          "pace": 100.7, "off_rtg": 107.9, "def_rtg": 118.4, "last10": 2,
                          "home": (13, 23), "away": (7, 31), "rest_days": 2, "b2b": False, "injuries": []},
    "washington wizards": {"name": "Washington Wizards", "abbr": "WAS", "conf": "East", "elo": 1570,
                           "pace": 101.4, "off_rtg": 107.1, "def_rtg": 119.6, "last10": 1,
                           "home": (11, 25), "away": (6, 32), "rest_days": 1, "b2b": True, "injuries": []},
}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def find_tennis_player(query: str) -> dict[str, Any] | None:
    """Fuzzy-ish lookup: exact key, then surname, then substring."""
    if not query:
        return None
    q = query.strip().lower()
    if q in TENNIS_PLAYERS:
        return dict(TENNIS_PLAYERS[q])
    for key, prof in TENNIS_PLAYERS.items():
        if key.split()[-1] == q:
            return dict(prof)
    for key, prof in TENNIS_PLAYERS.items():
        if q in key or key in q:
            return dict(prof)
    return None


def find_nba_team(query: str) -> dict[str, Any] | None:
    if not query:
        return None
    q = query.strip().lower()
    aliases = {
        "sixers": "philadelphia 76ers", "76ers": "philadelphia 76ers",
        "blazers": "portland trail blazers", "dubs": "golden state warriors",
        "warriors": "golden state warriors", "lakers": "los angeles lakers",
        "clips": "la clippers", "wolves": "minnesota timberwolves",
        "cavs": "cleveland cavaliers", "knicks": "new york knicks",
        "nugs": "denver nuggets", "thunder": "oklahoma city thunder",
        "okc": "oklahoma city thunder", "celts": "boston celtics",
        "celtics": "boston celtics", "mavs": "dallas mavericks",
        "heat": "miami heat", "suns": "phoenix suns", "bucks": "milwaukee bucks",
        "spurs": "san antonio spurs", "nets": "brooklyn nets", "bulls": "chicago bulls",
        "hawks": "atlanta hawks", "magic": "orlando magic", "kings": "sacramento kings",
        "rockets": "houston rockets", "grizzlies": "memphis grizzlies",
        "pistons": "detroit pistons", "hornets": "charlotte hornets",
        "wizards": "washington wizards", "raptors": "toronto raptors",
        "jazz": "utah jazz", "pelicans": "new orleans pelicans",
    }
    if q in NBA_TEAMS:
        return dict(NBA_TEAMS[q])
    if q in aliases:
        return dict(NBA_TEAMS[aliases[q]])
    for key, prof in NBA_TEAMS.items():
        if prof["abbr"].lower() == q:
            return dict(prof)
    for key, prof in NBA_TEAMS.items():
        if q in key or key in q:
            return dict(prof)
    for key, prof in NBA_TEAMS.items():
        if q.split()[-1] == key.split()[-1]:
            return dict(prof)
    return None


def tennis_player_names() -> list[str]:
    return sorted(p["name"] for p in TENNIS_PLAYERS.values())


def nba_team_names() -> list[str]:
    return sorted(p["name"] for p in NBA_TEAMS.values())


# ---------------------------------------------------------------------------
# Deterministic calibration history
# ---------------------------------------------------------------------------
def _tennis_features(a: dict[str, Any], b: dict[str, Any], surface: str) -> dict[str, float]:
    def surface_elo(p: dict[str, Any]) -> float:
        return float(p.get(f"elo_{surface}", p["elo"]))

    elo_diff = (a["elo"] - b["elo"]) / 400.0
    surf_diff = (surface_elo(a) - surface_elo(b)) / 400.0
    form_diff = (sum(a["form"]) - sum(b["form"])) / 10.0
    h2h = a.get("h2h", {}).get(b["name"].lower())
    if h2h and sum(h2h) > 0:
        h2h_diff = (h2h[0] - h2h[1]) / sum(h2h)
    else:
        h2h_diff = 0.0
    fatigue_diff = (b["matches_7d"] - a["matches_7d"]) / 7.0
    injury_diff = (0.0 if not a.get("injury") else -0.5) - (0.0 if not b.get("injury") else -0.5)
    serve_diff = (a["serve_pct"] - b["serve_pct"]) * 10
    return {
        "elo_diff": elo_diff,
        "surface_diff": surf_diff,
        "form_diff": form_diff,
        "h2h_diff": h2h_diff,
        "fatigue_diff": fatigue_diff,
        "injury_diff": injury_diff,
        "serve_diff": serve_diff,
    }


@lru_cache(maxsize=1)
def tennis_history(n_matches: int = 2400, seed: int = 20260101) -> pd.DataFrame:
    """Seeded, structured match history for calibration/backtest."""
    rng = np.random.default_rng(seed)
    keys = list(TENNIS_PLAYERS)
    surfaces = ["hard", "clay", "grass"]
    rows = []
    for _ in range(n_matches):
        i, j = rng.choice(len(keys), size=2, replace=False)
        a, b = TENNIS_PLAYERS[keys[i]], TENNIS_PLAYERS[keys[j]]
        surface = surfaces[int(rng.integers(0, 3))]
        f = _tennis_features(a, b, surface)

        z = (0.95 * f["elo_diff"] + 0.55 * f["surface_diff"] + 0.30 * f["form_diff"]
             + 0.22 * f["h2h_diff"] + 0.35 * f["fatigue_diff"] + 0.40 * f["injury_diff"]
             + 0.08 * f["serve_diff"])
        p = float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.03, 0.97))
        y = int(rng.random() < p)

        rows.append({
            "sport": "tennis",
            "player_a": a["name"], "player_b": b["name"], "surface": surface,
            "a_won": y, **f,
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def nba_history(n_games: int = 1800, seed: int = 20260102) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = list(NBA_TEAMS)
    rows = []
    for _ in range(n_games):
        i, j = rng.choice(len(keys), size=2, replace=False)
        h, a = NBA_TEAMS[keys[i]], NBA_TEAMS[keys[j]]

        elo_diff = (h["elo"] - a["elo"]) / 400.0
        net_diff = ((h["off_rtg"] - h["def_rtg"]) - (a["off_rtg"] - a["def_rtg"])) / 10.0
        form_diff = (h["last10"] - a["last10"]) / 10.0
        rest_diff = (h["rest_days"] - a["rest_days"]) / 3.0
        b2b_pen = (-0.35 if h["b2b"] else 0.0) - (-0.35 if a["b2b"] else 0.0)
        inj = (-0.25 * len(h["injuries"])) - (-0.25 * len(a["injuries"]))
        pace = (h["pace"] + a["pace"]) / 2

        z = (1.05 * elo_diff + 0.60 * net_diff + 0.25 * form_diff
             + 0.30 * rest_diff + 0.45 * b2b_pen + 0.35 * inj + 0.18)
        p = float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.05, 0.95))
        y = int(rng.random() < p)

        exp_total = pace * ((h["off_rtg"] + a["def_rtg"]) + (a["off_rtg"] + h["def_rtg"])) / 200.0
        actual_total = exp_total + rng.normal(0, 12.5)

        rows.append({
            "sport": "nba", "home": h["name"], "away": a["name"], "home_won": y,
            "elo_diff": elo_diff, "net_diff": net_diff, "form_diff": form_diff,
            "rest_diff": rest_diff, "b2b_diff": b2b_pen, "injury_diff": inj,
            "pace": pace, "exp_total": exp_total, "actual_total": float(actual_total),
        })
    return pd.DataFrame(rows)


def calibration_frame(sport: str) -> pd.DataFrame:
    return tennis_history() if sport == "tennis" else nba_history()
