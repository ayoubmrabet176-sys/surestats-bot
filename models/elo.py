"""
ELO engine + probability calibration.

Two layers:

1. ELO maths - classic logistic ELO with surface-specific ratings for tennis
   and a home-court bonus for NBA. Ratings are updated with the standard
   K-factor rule and persisted in SQLite so the bot keeps learning forever.

2. Calibration - each signal (ELO, form, surface/home, H2H, fatigue/injury)
   is converted to its own probability, then blended with the configured
   weights. The blended logit is passed through a scikit-learn
   LogisticRegression (Platt scaling) fitted on the historical results, so
   the final number is a calibrated probability rather than a raw score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import WEIGHTS, settings
from data import datasets
from utils.logging_conf import get_logger

log = get_logger("elo")

LN10_400 = math.log(10) / 400.0  # 0.005756


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def expected_score(elo_a: float, elo_b: float, scale: float | None = None) -> float:
    """Standard ELO expectation for player/team A."""
    scale = scale or settings.elo_scale
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / scale))


def update_rating(elo: float, expected: float, actual: float, k: float) -> float:
    return elo + k * (actual - expected)


@dataclass
class EloBook:
    """In-memory ELO table with optional persistence hooks."""

    k: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, key: str, default: float = 1500.0) -> float:
        return self.ratings.get(key, default)

    def apply_result(self, winner: str, loser: str, margin: float = 1.0) -> tuple[float, float]:
        ra, rb = self.rating(winner), self.rating(loser)
        exp_a = expected_score(ra, rb)
        mov = max(1.0, math.log1p(abs(margin)) + 1.0)  # margin-of-victory multiplier
        new_a = update_rating(ra, exp_a, 1.0, self.k * mov)
        new_b = update_rating(rb, 1 - exp_a, 0.0, self.k * mov)
        self.ratings[winner], self.ratings[loser] = new_a, new_b
        return new_a, new_b

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in sorted(self.ratings.items(), key=lambda kv: -kv[1])}


class TennisElo(EloBook):
    """ELO with a separate rating per surface, blended toward the overall rating."""

    SURFACE_WEIGHT = 0.65  # how much the surface rating dominates

    def __init__(self, k: float | None = None):
        super().__init__(k=k or settings.elo_k_tennis)
        self.surface: dict[str, dict[str, float]] = {"hard": {}, "clay": {}, "grass": {}}
        for name, prof in datasets.TENNIS_PLAYERS.items():
            self.ratings[name] = float(prof["elo"])
            for s in self.surface:
                self.surface[s][name] = float(prof.get(f"elo_{s}", prof["elo"]))

    def surface_rating(self, name: str, surface: str) -> float:
        base = self.rating(name)
        s = (surface or "hard").lower()
        surf = self.surface.get(s, {}).get(name, base)
        return self.SURFACE_WEIGHT * surf + (1 - self.SURFACE_WEIGHT) * base

    def blend_rating(self, name: str, surface: str) -> float:
        return self.surface_rating(name, surface)


class NBAElo(EloBook):
    def __init__(self, k: float | None = None):
        super().__init__(k=k or settings.elo_k_nba)
        for name, prof in datasets.NBA_TEAMS.items():
            self.ratings[name] = float(prof["elo"])


# ---------------------------------------------------------------------------
# component probabilities
# ---------------------------------------------------------------------------
def _form_score(results: list[int]) -> float:
    """Weighted recent form in [-1, 1]; recent matches count more."""
    if not results:
        return 0.0
    w = np.linspace(1.6, 0.6, len(results))
    vals = np.array(results, dtype=float) * 2 - 1
    return float(np.dot(vals, w) / w.sum())


def tennis_components(a: dict[str, Any], b: dict[str, Any], surface: str) -> dict[str, float]:
    elo = TennisElo()
    ra = elo.blend_rating(a["name"].lower(), surface)
    rb = elo.blend_rating(b["name"].lower(), surface)

    p_elo = sigmoid((ra - rb) * LN10_400 * 1.15)
    p_surface = sigmoid((elo.surface_rating(a["name"].lower(), surface)
                         - elo.surface_rating(b["name"].lower(), surface)) * LN10_400 * 1.35)

    fa, fb = _form_score(a["form"]), _form_score(b["form"])
    p_form = sigmoid((fa - fb) * 1.6)

    h2h = a.get("h2h", {}).get(b["name"].lower())
    if h2h and sum(h2h) > 0:
        wins, losses = h2h
        rate = (wins + 1.0) / (wins + losses + 2.0)  # Laplace smoothing
        p_h2h = sigmoid((rate - 0.5) * 3.2)
    else:
        p_h2h = 0.5

    def penalty(p: dict[str, Any]) -> float:
        fatigue = max(0.0, p.get("matches_7d", 0) - 4) * 0.16
        injury = 0.0
        note = (p.get("injury") or "").lower()
        if note:
            if "out" in note:
                injury = 0.9
            elif "doubtful" in note:
                injury = 0.55
            elif "questionable" in note:
                injury = 0.35
            else:
                injury = 0.18
        return fatigue + injury

    pa, pb = penalty(a), penalty(b)
    p_fi = sigmoid((pb - pa) * 2.2)
    return {"elo": p_elo, "form": p_form, "surface_or_home": p_surface,
            "h2h": p_h2h, "fatigue_injury": p_fi}


def nba_components(home: dict[str, Any], away: dict[str, Any]) -> dict[str, float]:
    elo = NBAElo()
    rh = elo.rating(home["name"].lower()) + settings.home_advantage_nba * 0.5
    ra = elo.rating(away["name"].lower())
    p_elo = sigmoid((rh - ra) * LN10_400 * 1.2)

    net_h = home["off_rtg"] - home["def_rtg"]
    net_a = away["off_rtg"] - away["def_rtg"]
    p_home = sigmoid((net_h - net_a) / 6.0 + 0.22)  # +0.22 = home court

    p_form = sigmoid((home["last10"] - away["last10"]) / 10.0 * 2.4)

    h2h = home.get("h2h", {}).get(away["name"].lower())
    if h2h and sum(h2h) > 0:
        rate = (h2h[0] + 1.0) / (sum(h2h) + 2.0)
        p_h2h = sigmoid((rate - 0.5) * 3.0)
    else:
        p_h2h = 0.5

    def penalty(t: dict[str, Any]) -> float:
        rest = t.get("rest_days", 2)
        pen = 0.0
        if t.get("b2b"):
            pen += 0.45
        if rest == 0:
            pen += 0.25
        elif rest >= 3:
            pen -= 0.10  # well rested bonus
        pen += 0.30 * len(t.get("injuries", []))
        return pen

    p_fi = sigmoid((penalty(away) - penalty(home)) * 1.8)
    return {"elo": p_elo, "form": p_form, "surface_or_home": p_home,
            "h2h": p_h2h, "fatigue_injury": p_fi}


def blend(components: dict[str, float]) -> float:
    """Weighted blend of component probabilities (weights sum to 1)."""
    total_w = sum(WEIGHTS.values())
    p = sum(WEIGHTS[k] * components[k] for k in WEIGHTS) / total_w
    return float(min(max(p, 0.01), 0.99))


# ---------------------------------------------------------------------------
# Platt calibration (scikit-learn)
# ---------------------------------------------------------------------------
# The multipliers below mirror the live component formulas exactly so the
# calibrator is fitted on the same feature definition used at inference time.
TENNIS_ELO_MULT = 400 * LN10_400 * 1.15      # 2.648
TENNIS_SURF_MULT = 400 * LN10_400 * 1.35     # 3.108
NBA_ELO_MULT = 400 * LN10_400 * 1.2          # 2.763


def _tennis_training_matrix() -> tuple[np.ndarray, np.ndarray]:
    df = datasets.tennis_history()
    X, y = [], []
    for row in df.itertuples(index=False):
        comp = {
            "elo": sigmoid(row.elo_diff * TENNIS_ELO_MULT),
            "form": sigmoid(row.form_diff * 1.6),
            "surface_or_home": sigmoid(row.surface_diff * TENNIS_SURF_MULT),
            "h2h": sigmoid(row.h2h_diff * 3.2),
            "fatigue_injury": sigmoid((row.fatigue_diff * 0.5 + row.injury_diff * 0.5) * 2.2),
        }
        X.append([logit(blend(comp))])
        y.append(int(row.a_won))
    return np.array(X), np.array(y)


def _nba_training_matrix() -> tuple[np.ndarray, np.ndarray]:
    df = datasets.nba_history()
    X, y = [], []
    for row in df.itertuples(index=False):
        comp = {
            "elo": sigmoid(row.elo_diff * NBA_ELO_MULT),
            "form": sigmoid(row.form_diff * 2.4),
            "surface_or_home": sigmoid(row.net_diff / 6.0 + 0.22),
            "h2h": 0.5,
            "fatigue_injury": sigmoid((row.rest_diff * 0.35 + row.b2b_diff * 0.45
                                       + row.injury_diff * 0.35) * 1.8),
        }
        X.append([logit(blend(comp))])
        y.append(int(row.home_won))
    return np.array(X), np.array(y)


@lru_cache(maxsize=2)
def calibrator(sport: str) -> LogisticRegression:
    X, y = _tennis_training_matrix() if sport == "tennis" else _nba_training_matrix()
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    log.info("calibrator fitted for %s on %s matches (coef=%.3f, intercept=%.3f)",
             sport, len(y), model.coef_[0][0], model.intercept_[0])
    return model


def calibrated_probability(sport: str, components: dict[str, float]) -> float:
    p = blend(components)
    model = calibrator(sport)
    return float(model.predict_proba(np.array([[logit(p)]]))[0, 1])


def calibration_summary(sport: str) -> dict[str, float]:
    m = calibrator(sport)
    return {"coef": float(m.coef_[0][0]), "intercept": float(m.intercept_[0])}
