"""
The core analysis engine.

Given a sport + match, it:
  1. builds normalised profiles (live API where available, bootstrap otherwise)
  2. computes per-signal probabilities and blends them (ELO 35 / form 30 /
     surface-or-home 20 / H2H 10 / fatigue-injury 5)
  3. calibrates the blend with scikit-learn
  4. prices every selection against the best available bookmaker price
  5. returns value %, confidence, risk and a human-readable rationale
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from api import nba_api, odds_api, tennis_api
from config import WEIGHTS, settings
from data import db
from models import elo as elo_mod
from models.totals import totals_market
from utils.logging_conf import get_logger

log = get_logger("engine")


# ---------------------------------------------------------------------------
# result objects
# ---------------------------------------------------------------------------
@dataclass
class Selection:
    selection: str
    market: str
    true_prob: float
    book_prob: float
    best_odds: float
    best_book: str
    avg_odds: float
    value_pct: float
    edge_pct: float
    signal: str
    confidence: float
    risk: str
    reasoning: list[str] = field(default_factory=list)
    line: float | None = None

    @property
    def kelly_stake(self) -> float:
        """Fraction of bankroll for a quarter-Kelly stake."""
        b = self.best_odds - 1
        if b <= 0:
            return 0.0
        f = (self.true_prob * b - (1 - self.true_prob)) / b
        return round(max(0.0, f) * 0.25, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection, "market": self.market,
            "true_prob": round(self.true_prob, 4), "book_prob": round(self.book_prob, 4),
            "best_odds": round(self.best_odds, 3), "best_book": self.best_book,
            "avg_odds": round(self.avg_odds, 3), "value_pct": round(self.value_pct, 2),
            "edge_pct": round(self.edge_pct, 2), "signal": self.signal,
            "confidence": round(self.confidence, 3), "risk": self.risk,
            "reasoning": self.reasoning, "line": self.line,
            "kelly_stake": self.kelly_stake,
        }


@dataclass
class Analysis:
    sport: str
    match: str
    event_time: str
    surface: str | None
    selections: list[Selection]
    market: dict[str, Any]
    profiles: dict[str, Any]
    components: dict[str, float]
    totals: dict[str, Any] | None = None
    data_quality: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def best(self) -> Selection | None:
        return max(self.selections, key=lambda s: s.value_pct) if self.selections else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sport": self.sport, "match": self.match, "event_time": self.event_time,
            "surface": self.surface, "created_at": self.created_at,
            "selections": [s.to_dict() for s in self.selections],
            "totals": self.totals, "components": self.components,
            "data_quality": self.data_quality, "market_source": self.market.get("source"),
            "market_note": self.market.get("note"), "book_count": self.market.get("book_count"),
            "profiles": self.profiles,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def signal_for(value_pct: float) -> str:
    if value_pct >= settings.value_green:
        return "VALUE BET - GREEN"
    if value_pct >= settings.value_yellow:
        return "LOW VALUE - YELLOW"
    return "NO VALUE - RED"


def risk_for(confidence: float, value_pct: float, odds: float) -> str:
    score = (1 - confidence) * 2.0
    score += 0.6 if odds >= 3.5 else 0.3 if odds >= 2.2 else 0.0
    score += 0.5 if value_pct >= 20 else 0.25 if value_pct >= 12 else 0.0
    if score >= 1.5:
        return "High"
    if score >= 0.8:
        return "Medium"
    return "Low"


def _confidence(market: dict[str, Any], profiles: dict[str, Any], n_form: int) -> float:
    """0-1 score built from data completeness, not from the size of the edge."""
    score = 0.30
    books = int(market.get("book_count") or 0)
    score += min(0.25, books / 24 * 0.25)
    if market.get("source") == "live":
        score += 0.10
    live_sources = sum(1 for p in profiles.values() if "bootstrap" not in str(p.get("source", "")))
    score += min(0.15, live_sources * 0.075)
    score += min(0.12, n_form / 10 * 0.12)
    return float(min(0.97, max(0.25, score)))


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# tennis
# ---------------------------------------------------------------------------
def analyze_tennis(match: str, event_time: str = "", surface: str = "hard") -> Analysis:
    parts = [p.strip() for p in match.replace(" vs ", " v ").split(" v ")]
    if len(parts) != 2:
        raise ValueError("Tennis match must be 'Player A vs Player B'")
    name_a, name_b = parts

    a = tennis_api.player_profile(name_a)
    b = tennis_api.player_profile(name_b)
    a["form"] = tennis_api.recent_form(name_a)
    b["form"] = tennis_api.recent_form(name_b)
    h2h = tennis_api.head_to_head(name_a, name_b)
    if h2h:
        a["h2h"] = {b["name"].lower(): h2h}
        b["h2h"] = {a["name"].lower(): [h2h[1], h2h[0]]}

    surface = (surface or "hard").lower()
    comp = elo_mod.tennis_components(a, b, surface)
    p_a = elo_mod.calibrated_probability("tennis", comp)
    p_b = 1 - p_a

    market = odds_api.get_market("tennis", match, a["name"], b["name"], p_a, p_b)
    quotes = market["moneyline"]

    book_p_a = odds_api.devigged_probability(quotes, "A", "B") or p_a
    book_p_b = 1 - book_p_a

    confidence = _confidence(market, {"a": a, "b": b}, min(len(a["form"]), len(b["form"])))

    sel_a = _build_selection(
        selection=a["name"], market_name="Match Winner (H2H)", true_prob=p_a,
        book_prob=book_p_a, quotes=quotes, sel_key="A", confidence=confidence,
        reasoning=_tennis_reasoning(a, b, surface, p_a, book_p_a, quotes, sel_key="A"),
    )
    sel_b = _build_selection(
        selection=b["name"], market_name="Match Winner (H2H)", true_prob=p_b,
        book_prob=book_p_b, quotes=quotes, sel_key="B", confidence=confidence,
        reasoning=_tennis_reasoning(a, b, surface, p_b, book_p_b, quotes, sel_key="B"),
    )

    analysis = Analysis(
        sport="tennis", match=f"{a['name']} vs {b['name']}", event_time=event_time,
        surface=surface, selections=[sel_a, sel_b], market=market,
        profiles={"a": a, "b": b}, components=comp,
        data_quality={
            "book_count": market.get("book_count"),
            "source": market.get("source"),
            "profile_sources": {"a": a.get("source"), "b": b.get("source")},
            "form_sample": {"a": len(a["form"]), "b": len(b["form"])},
        },
    )
    _persist(analysis)
    return analysis


def _tennis_reasoning(a: dict[str, Any], b: dict[str, Any], surface: str,
                      p: float, book_p: float,
                      quotes: list[dict[str, Any]], sel_key: str) -> list[str]:
    pick, opp = (a, b) if sel_key == "A" else (b, a)
    pick_surface_elo = pick.get(f"elo_{surface}", pick["elo"])
    opp_surface_elo = opp.get(f"elo_{surface}", opp["elo"])
    out = [
        f"{pick['name']} rates {pick_surface_elo:.0f} on {surface} vs {opp['name']} at "
        f"{opp_surface_elo:.0f} (overall ELO {pick['elo']:.0f} vs {opp['elo']:.0f}).",
        f"Form: {pick['name']} {sum(pick['form'])}/{len(pick['form'])} in last {len(pick['form'])} "
        f"vs {opp['name']} {sum(opp['form'])}/{len(opp['form'])}.",
    ]
    h2h = pick.get("h2h", {}).get(opp["name"].lower())
    if h2h:
        out.append(f"Head-to-head: {pick['name']} stands {h2h[0]}-{h2h[1]} against {opp['name']}.")
    else:
        out.append("No meaningful head-to-head sample - H2H held at neutral 50%.")
    out.append(
        f"Workload: {pick['name']} has played {pick['matches_7d']} matches in the last 7 days "
        f"vs {opp['matches_7d']} for {opp['name']}."
    )
    if pick.get("injury"):
        out.append(f"Injury flag on {pick['name']}: {pick['injury']}.")
    if opp.get("injury"):
        out.append(f"Opponent injury flag: {opp['name']} - {opp['injury']}.")
    out.append(
        f"Serve quality: {_fmt_pct(pick['serve_pct'])} first serves in, "
        f"{_fmt_pct(pick['bp_saved'])} break points saved."
    )
    out.append(
        f"Model probability {_fmt_pct(p)} vs market {_fmt_pct(book_p)} "
        f"({(p - book_p) * 100:+.1f} pts). Signal weights: ELO {WEIGHTS['elo']:.0%}, "
        f"form {WEIGHTS['form']:.0%}, surface {WEIGHTS['surface_or_home']:.0%}, "
        f"H2H {WEIGHTS['h2h']:.0%}, fatigue/injury {WEIGHTS['fatigue_injury']:.0%}."
    )
    best, book = odds_api.best_price(quotes, sel_key)
    if best:
        out.append(f"Best price {best:.2f} at {book}.")
    return out


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------
def analyze_nba(match: str, event_time: str = "", home_team: str | None = None) -> Analysis:
    parts = [p.strip() for p in match.replace(" vs ", " v ").split(" v ")]
    if len(parts) != 2:
        raise ValueError("NBA match must be 'Home vs Away'")

    if home_team:
        home_name = home_team
        away_name = parts[1] if parts[0].lower() == home_team.lower() else parts[0]
    else:
        home_name, away_name = parts[0], parts[1]

    home = nba_api.team_profile(home_name)
    away = nba_api.team_profile(away_name)
    home["injuries"] = nba_api.injury_report(home_name)
    away["injuries"] = nba_api.injury_report(away_name)

    comp = elo_mod.nba_components(home, away)
    p_home = elo_mod.calibrated_probability("nba", comp)
    p_away = 1 - p_home

    totals = totals_market(home, away)
    market = odds_api.get_market(
        "nba", match, home["name"], away["name"], p_home, p_away,
        totals_line=totals["line"], p_over=totals["p_over"],
    )
    quotes = market["moneyline"]
    book_p_home = odds_api.devigged_probability(quotes, "A", "B") or p_home
    book_p_away = 1 - book_p_home

    confidence = _confidence(market, {"home": home, "away": away}, 10)

    sel_home = _build_selection(
        selection=f"{home['name']} (ML)", market_name="Moneyline", true_prob=p_home,
        book_prob=book_p_home, quotes=quotes, sel_key="A", confidence=confidence,
        reasoning=_nba_reasoning(home, away, comp, p_home, book_p_home, quotes, "A"),
    )
    sel_away = _build_selection(
        selection=f"{away['name']} (ML)", market_name="Moneyline", true_prob=p_away,
        book_prob=book_p_away, quotes=quotes, sel_key="B", confidence=confidence,
        reasoning=_nba_reasoning(home, away, comp, p_away, book_p_away, quotes, "B"),
    )

    selections = [sel_home, sel_away]

    t_quotes = market.get("totals") or []
    if t_quotes:
        line = market.get("line") or totals["line"]
        totals["line"] = line
        book_p_over = odds_api.devigged_probability(t_quotes, "Over", "Under") or totals["p_over"]
        selections.append(_build_selection(
            selection=f"Over {line}", market_name="Total Points", true_prob=totals["p_over"],
            book_prob=book_p_over, quotes=t_quotes, sel_key="Over", confidence=confidence,
            reasoning=[
                f"Model expects {totals['expected_total']} points (pace {totals['pace']}, "
                f"combined OFF {totals['combined_off']} / DEF {totals['combined_def']}).",
                f"Line is {line}; model edge {totals['edge_points']:+.1f} points.",
                f"Residual sigma {totals['sigma']} pts from historical totals errors.",
            ],
            line=line,
        ))
        selections.append(_build_selection(
            selection=f"Under {line}", market_name="Total Points", true_prob=totals["p_under"],
            book_prob=1 - book_p_over, quotes=t_quotes, sel_key="Under", confidence=confidence,
            reasoning=[
                f"Model expects {totals['expected_total']} points vs a line of {line} "
                f"({totals['edge_points']:+.1f} points).",
                f"Combined pace {totals['pace']} with injury/rest drag priced in.",
            ],
            line=line,
        ))

    analysis = Analysis(
        sport="nba", match=f"{home['name']} vs {away['name']}", event_time=event_time,
        surface=None, selections=selections, market=market,
        profiles={"home": home, "away": away}, components=comp, totals=totals,
        data_quality={
            "book_count": market.get("book_count"),
            "source": market.get("source"),
            "profile_sources": {"home": home.get("source"), "away": away.get("source")},
            "injuries": {"home": home["injuries"], "away": away["injuries"]},
        },
    )
    _persist(analysis)
    return analysis


def _nba_reasoning(home: dict[str, Any], away: dict[str, Any], comp: dict[str, float],
                   p: float, book_p: float, quotes: list[dict[str, Any]], sel_key: str) -> list[str]:
    pick, opp = (home, away) if sel_key == "A" else (away, home)
    out = [
        f"{pick['name']} net rating {pick['off_rtg'] - pick['def_rtg']:+.1f} "
        f"(OFF {pick['off_rtg']}, DEF {pick['def_rtg']}) vs {opp['name']} "
        f"{opp['off_rtg'] - opp['def_rtg']:+.1f} (OFF {opp['off_rtg']}, DEF {opp['def_rtg']}).",
        f"Last 10: {pick['name']} {pick['last10']}-{10 - pick['last10']} vs "
        f"{opp['name']} {opp['last10']}-{10 - opp['last10']}.",
        f"Rest: {pick['name']} {pick['rest_days']} day(s){' (back-to-back)' if pick['b2b'] else ''} "
        f"vs {opp['name']} {opp['rest_days']} day(s){' (back-to-back)' if opp['b2b'] else ''}.",
        f"Home/away: {home['name']} {home['home'][0]}-{home['home'][1]} at home, "
        f"{away['name']} {away['away'][0]}-{away['away'][1]} on the road.",
    ]
    if pick["injuries"]:
        out.append(f"Injury report ({pick['name']}): " + "; ".join(pick["injuries"]))
    else:
        out.append(f"No listed injuries for {pick['name']}.")
    if opp["injuries"]:
        out.append(f"Opponent injuries ({opp['name']}): " + "; ".join(opp["injuries"]))
    out.append(
        f"Model {_fmt_pct(p)} vs market {_fmt_pct(book_p)} ({(p - book_p) * 100:+.1f} pts). "
        f"ELO {comp['elo']:.1%}, form {comp['form']:.1%}, home {comp['surface_or_home']:.1%}, "
        f"H2H {comp['h2h']:.1%}, rest/injury {comp['fatigue_injury']:.1%}."
    )
    best, book = odds_api.best_price(quotes, sel_key)
    if best:
        out.append(f"Best price {best:.2f} at {book}.")
    return out


# ---------------------------------------------------------------------------
# shared selection builder
# ---------------------------------------------------------------------------
def _build_selection(*, selection: str, market_name: str, true_prob: float, book_prob: float,
                     quotes: list[dict[str, Any]], sel_key: str, confidence: float,
                     reasoning: list[str], line: float | None = None) -> Selection:
    best_odds, best_book = odds_api.best_price(quotes, sel_key)
    avg_odds = odds_api.average_price(quotes, sel_key)
    if best_odds <= 1.0:
        best_odds = round(1.0 / max(true_prob, 0.01), 3)
        best_book = "model fair price"
        avg_odds = best_odds

    value_pct = (true_prob * best_odds - 1) * 100
    edge_pct = (true_prob - book_prob) * 100
    return Selection(
        selection=selection, market=market_name, true_prob=true_prob, book_prob=book_prob,
        best_odds=best_odds, best_book=best_book, avg_odds=avg_odds,
        value_pct=value_pct, edge_pct=edge_pct, signal=signal_for(value_pct),
        confidence=confidence, risk=risk_for(confidence, value_pct, best_odds),
        reasoning=reasoning, line=line,
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def _persist(analysis: Analysis) -> None:
    try:
        for s in analysis.selections:
            db.save_analysis({
                "sport": analysis.sport, "match": analysis.match,
                "event_time": analysis.event_time, "selection": s.selection,
                "market": s.market, "true_prob": s.true_prob, "book_prob": s.book_prob,
                "best_odds": s.best_odds, "best_book": s.best_book,
                "value_pct": s.value_pct, "edge_pct": s.edge_pct, "signal": s.signal,
                "confidence": s.confidence, "risk": s.risk,
                "reasoning": " ".join(s.reasoning),
                "payload": {"line": s.line, "kelly": s.kelly_stake,
                            "components": analysis.components,
                            "market_source": analysis.market.get("source")},
            })
        db.save_odds_snapshot(analysis.sport, analysis.match, "moneyline", analysis.market["moneyline"])
        if analysis.market.get("totals"):
            db.save_odds_snapshot(analysis.sport, analysis.match, "totals", analysis.market["totals"])
    except Exception as exc:  # persistence must never break analysis
        log.exception("failed to persist analysis: %s", exc)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def analyze(sport: str, match: str, event_time: str = "", **kwargs: Any) -> Analysis:
    sport = (sport or "").strip().lower()
    if sport not in {"tennis", "nba"}:
        raise ValueError("sport must be 'tennis' or 'nba'")
    log.info("analyze %s | %s", sport, match)
    if sport == "tennis":
        return analyze_tennis(match, event_time, kwargs.get("surface", "hard"))
    return analyze_nba(match, event_time, kwargs.get("home_team"))
