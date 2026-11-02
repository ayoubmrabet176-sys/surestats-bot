"""
Historical validation: accuracy, Brier score, log loss and a simulated ROI
curve for the value strategy.

The backtest replays the calibration history through the *same* component
blend + calibrator used live, prices each bet at a simulated best-of-market
odds (fair odds plus a small margin), and only stakes when value >= 3%.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from config import settings
from data import datasets, db
from models import elo as elo_mod
from utils.logging_conf import get_logger

log = get_logger("backtest")


def _tennis_rows() -> list[dict[str, Any]]:
    df = datasets.tennis_history()
    rows = []
    for r in df.itertuples(index=False):
        comp = {
            "elo": elo_mod.sigmoid(r.elo_diff * elo_mod.TENNIS_ELO_MULT),
            "form": elo_mod.sigmoid(r.form_diff * 1.6),
            "surface_or_home": elo_mod.sigmoid(r.surface_diff * elo_mod.TENNIS_SURF_MULT),
            "h2h": elo_mod.sigmoid(r.h2h_diff * 3.2),
            "fatigue_injury": elo_mod.sigmoid((r.fatigue_diff * 0.5 + r.injury_diff * 0.5) * 2.2),
        }
        rows.append({"components": comp, "won": int(r.a_won), "label": f"{r.player_a} v {r.player_b}"})
    return rows


def _nba_rows() -> list[dict[str, Any]]:
    df = datasets.nba_history()
    rows = []
    for r in df.itertuples(index=False):
        comp = {
            "elo": elo_mod.sigmoid(r.elo_diff * elo_mod.NBA_ELO_MULT),
            "form": elo_mod.sigmoid(r.form_diff * 2.4),
            "surface_or_home": elo_mod.sigmoid(r.net_diff / 6.0 + 0.22),
            "h2h": 0.5,
            "fatigue_injury": elo_mod.sigmoid((r.rest_diff * 0.35 + r.b2b_diff * 0.45
                                               + r.injury_diff * 0.35) * 1.8),
        }
        rows.append({"components": comp, "won": int(r.home_won), "label": f"{r.home} v {r.away}"})
    return rows


def run_backtest(sport: str = "tennis", min_value_pct: float = 3.0) -> dict[str, Any]:
    rows = _tennis_rows() if sport == "tennis" else _nba_rows()
    model = elo_mod.calibrator(sport)

    logits = np.array([[elo_mod.logit(elo_mod.blend(r["components"]))] for r in rows])
    probs = model.predict_proba(logits)[:, 1]
    y = np.array([r["won"] for r in rows])

    # simulated market: fair odds with a 4.5% margin, plus book noise
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.02, size=len(probs))
    market_p = np.clip(probs + noise, 0.02, 0.98)
    best_odds = 1.0 / (market_p * (1 + 0.045 / 2))
    other_odds = 1.0 / ((1 - market_p) * (1 + 0.045 / 2))

    value_a = probs * best_odds - 1
    value_b = (1 - probs) * other_odds - 1

    bet_a = value_a >= value_b
    stake_prob = np.where(bet_a, probs, 1 - probs)
    odds_used = np.where(bet_a, best_odds, other_odds)
    won = np.where(bet_a, y == 1, y == 0)
    best_value = np.maximum(value_a, value_b)
    take = best_value >= (min_value_pct / 100)

    pnl = np.where(take, np.where(won, odds_used - 1, -1.0), 0.0)
    n_bets = int(take.sum())
    roi = float(pnl.sum() / n_bets * 100) if n_bets else 0.0

    pred = (probs >= 0.5).astype(int)
    accuracy = float((pred == y).mean())
    brier = float(np.mean((probs - y) ** 2))
    eps = 1e-9
    log_loss = float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))

    buckets = []
    for lo in np.arange(0.3, 1.0, 0.1):
        m = (probs >= lo) & (probs < lo + 0.1)
        if m.sum() >= 5:
            buckets.append({
                "bucket": f"{int(lo*100)}-{int((lo+0.1)*100)}%",
                "n": int(m.sum()),
                "predicted": round(float(probs[m].mean()), 4),
                "actual": round(float(y[m].mean()), 4),
            })

    curve, cum = [], 0.0
    for i in range(len(pnl)):
        if take[i]:
            cum += pnl[i]
            curve.append({"bet": len(curve) + 1, "cum_profit": round(cum, 3)})

    result = {
        "sport": sport, "model_version": "1.0.0",
        "n_bets": n_bets, "n_matches": len(y),
        "accuracy": round(accuracy, 4), "brier": round(brier, 4),
        "log_loss": round(log_loss, 4), "roi_pct": round(roi, 2),
        "detail": {
            "calibration": buckets,
            "profit_curve": curve,
            "hit_rate_bets": round(float(won[take].mean()), 4) if n_bets else 0.0,
            "avg_odds": round(float(odds_used[take].mean()), 3) if n_bets else 0.0,
            "avg_value_pct": round(float(best_value[take].mean() * 100), 2) if n_bets else 0.0,
        },
    }
    log.info("backtest %s: n=%s acc=%.3f brier=%.3f roi=%.1f%%",
             sport, n_bets, accuracy, brier, roi)
    return result


def run_and_store(sport: str = "tennis", min_value_pct: float = 3.0) -> dict[str, Any]:
    result = run_backtest(sport, min_value_pct)
    try:
        db.save_backtest(result)
    except Exception as exc:
        log.warning("could not store backtest: %s", exc)
    return result


def get_backtest(sport: str = "tennis") -> dict[str, Any]:
    """Stored backtest if present, otherwise compute and store one."""
    stored = db.latest_backtest(sport)
    if stored and stored.get("detail", {}).get("profit_curve") is not None:
        return stored
    return run_and_store(sport, settings.value_yellow)
