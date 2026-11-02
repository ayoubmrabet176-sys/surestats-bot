"""End-to-end smoke test: DB, engine, odds, totals, backtest, persistence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import db  # noqa: E402
from models import backtest, engine  # noqa: E402
from models import elo as elo_mod  # noqa: E402
from models.totals import totals_market  # noqa: E402


def main() -> int:
    db.init_db()
    print("DB initialised")

    print("calibration tennis:", elo_mod.calibration_summary("tennis"))
    print("calibration nba   :", elo_mod.calibration_summary("nba"))

    t = engine.analyze("tennis", "Djokovic vs Alcaraz", "2026-11-05T18:00Z", surface="hard")
    print(f"\nTENNIS: {t.match} ({t.surface})")
    for s in t.selections:
        print(f"  {s.selection:<22} true={s.true_prob:.1%} book={s.book_prob:.1%} "
              f"odds={s.best_odds:.2f}@{s.best_book:<12} value={s.value_pct:+.1f}% "
              f"{s.signal} conf={s.confidence:.2f} risk={s.risk}")
    print("  components:", {k: round(v, 3) for k, v in t.components.items()})
    print("  reasoning[0]:", t.selections[0].reasoning[0])

    n = engine.analyze("nba", "Lakers vs Warriors", "2026-11-05T02:00Z")
    print(f"\nNBA: {n.match}")
    for s in n.selections:
        print(f"  {s.selection:<28} true={s.true_prob:.1%} book={s.book_prob:.1%} "
              f"odds={s.best_odds:.2f}@{s.best_book:<12} value={s.value_pct:+.1f}% "
              f"{s.signal} conf={s.confidence:.2f} risk={s.risk}")
    print("  totals:", n.totals)

    bt_t = backtest.run_and_store("tennis")
    bt_n = backtest.run_and_store("nba")
    for bt in (bt_t, bt_n):
        print(f"\nBACKTEST {bt['sport']}: n_bets={bt['n_bets']} acc={bt['accuracy']:.3f} "
              f"brier={bt['brier']:.4f} logloss={bt['log_loss']:.4f} roi={bt['roi_pct']:+.1f}%")

    print("\nrecent analyses:", len(db.recent_analyses(20)))
    print("stats:", db.analysis_stats())
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
