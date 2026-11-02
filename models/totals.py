"""
NBA Over/Under model.

Expected total = average pace x (combined offensive rating vs combined
defensive rating) / 100, then adjusted for injuries and rest, with the
residual spread learned from the historical totals residuals (so the
probability is calibrated rather than guessed).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from data import datasets


@lru_cache(maxsize=1)
def residual_sigma() -> float:
    df = datasets.nba_history()
    resid = (df["actual_total"] - df["exp_total"]).to_numpy()
    return float(max(8.0, resid.std(ddof=1)))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def expected_total(home: dict[str, Any], away: dict[str, Any]) -> float:
    pace = (home["pace"] + away["pace"]) / 2.0
    home_pp100 = (home["off_rtg"] + away["def_rtg"]) / 2.0
    away_pp100 = (away["off_rtg"] + home["def_rtg"]) / 2.0
    base = pace * (home_pp100 + away_pp100) / 100.0

    # injury drag: each listed injury removes ~0.9 points of offence
    base -= 0.9 * len(home.get("injuries", [])) + 0.9 * len(away.get("injuries", []))
    # back-to-back legs are usually a touch slower / less efficient
    if home.get("b2b"):
        base -= 1.1
    if away.get("b2b"):
        base -= 1.1
    return float(base)


def totals_market(home: dict[str, Any], away: dict[str, Any], line: float | None = None) -> dict[str, Any]:
    mu = expected_total(home, away)
    sigma = residual_sigma()
    line = float(line if line is not None else round(mu * 2) / 2)  # nearest half point

    p_over = 1.0 - norm_cdf(line, mu, sigma)
    p_under = 1.0 - p_over

    return {
        "line": line,
        "expected_total": round(mu, 1),
        "sigma": round(sigma, 2),
        "p_over": p_over,
        "p_under": p_under,
        "edge_points": round(mu - line, 1),
        "pace": round((home["pace"] + away["pace"]) / 2, 1),
        "combined_off": round((home["off_rtg"] + away["off_rtg"]) / 2, 1),
        "combined_def": round((home["def_rtg"] + away["def_rtg"]) / 2, 1),
    }
