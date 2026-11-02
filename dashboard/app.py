"""
SureStats Bot - Streamlit dashboard.

Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import WEIGHTS, settings  # noqa: E402
from dashboard import components as ui  # noqa: E402
from dashboard.styles import CSS  # noqa: E402
from data import db  # noqa: E402
from data.datasets import nba_team_names, tennis_player_names  # noqa: E402
from models import backtest, engine  # noqa: E402
from models import elo as elo_mod  # noqa: E402
from utils.http import cache_clear  # noqa: E402
from utils.logging_conf import get_logger  # noqa: E402

log = get_logger("dashboard")

st.set_page_config(
    page_title="SureStats Bot - Tennis & NBA",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

db.init_db()

TENNIS_PRESETS = [
    "Novak Djokovic vs Carlos Alcaraz",
    "Jannik Sinner vs Alexander Zverev",
    "Carlos Alcaraz vs Jannik Sinner",
    "Iga Swiatek vs Aryna Sabalenka",
    "Coco Gauff vs Elena Rybakina",
    "Daniil Medvedev vs Taylor Fritz",
    "Casper Ruud vs Stefanos Tsitsipas",
    "Alex de Minaur vs Ben Shelton",
]
NBA_PRESETS = [
    "Los Angeles Lakers vs Golden State Warriors",
    "Boston Celtics vs New York Knicks",
    "Denver Nuggets vs Oklahoma City Thunder",
    "Milwaukee Bucks vs Cleveland Cavaliers",
    "Dallas Mavericks vs Minnesota Timberwolves",
    "Phoenix Suns vs LA Clippers",
    "Philadelphia 76ers vs Miami Heat",
    "Houston Rockets vs Memphis Grizzlies",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def api_status() -> list[tuple[str, bool, str]]:
    return [
        ("The Odds API", settings.has_odds_key, "ODDS_API_KEY"),
        ("API-Tennis", settings.has_tennis_key, "TENNIS_API_KEY"),
        ("balldontlie", settings.has_nba_key, "BALLDONTLIE_API_KEY"),
    ]


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def selections_frame(selections: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Selection": s["selection"],
            "Market": s["market"],
            "True %": round(s["true_prob"] * 100, 1),
            "Book %": round(s["book_prob"] * 100, 1),
            "Edge (pts)": round(s["edge_pct"], 1),
            "Best odds": s["best_odds"],
            "Bookmaker": s["best_book"],
            "Value %": round(s["value_pct"], 1),
            "Signal": s["signal"].replace(" - ", " · "),
            "Conf.": round(s["confidence"] * 100),
            "Risk": s["risk"],
            "¼-Kelly": f"{s['kelly_stake']*100:.1f}%",
        }
        for s in selections
    ])


def render_empty() -> None:
    st.markdown(
        f"""
        <div class="ss-card" style="padding:34px 30px;text-align:center;">
          <div style="font-size:38px;margin-bottom:8px;">🎾 🏀</div>
          <div style="font-size:19px;font-weight:700;margin-bottom:6px;">No analysis yet</div>
          <div style="color:#8b9ab5;font-size:14px;max-width:620px;margin:0 auto;line-height:1.65;">
            Pick a sport, choose a matchup (or type your own), set the start time and hit
            <b>Analyze</b>. The engine pulls both sides' profiles, blends five weighted signals,
            calibrates the probability with scikit-learn and prices every selection against the
            best of 24 bookmakers.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# results rendering
# ---------------------------------------------------------------------------
def render_analysis(a: engine.Analysis) -> None:
    data = a.to_dict()
    sels = data["selections"]
    best = max(sels, key=lambda s: s["value_pct"])

    st.markdown(
        ui.banner(
            best["signal"],
            f'{best["selection"]} &nbsp;·&nbsp; {best["value_pct"]:+.1f}% value',
            f'{a.match} &nbsp;|&nbsp; {best["market"]} &nbsp;|&nbsp; best price '
            f'<b>{best["best_odds"]:.2f}</b> at <b>{best["best_book"]}</b> &nbsp;|&nbsp; '
            f'true {fmt_pct(best["true_prob"])} vs market {fmt_pct(best["book_prob"])}',
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(ui.card("Best value", f'{best["value_pct"]:+.1f}%',
                            f'edge {best["edge_pct"]:+.1f} pts',
                            ui.signal_color(best["signal"])), unsafe_allow_html=True)
    with c2:
        st.markdown(ui.card("True probability", fmt_pct(best["true_prob"]),
                            "model, odds-independent", "#e8eef9"), unsafe_allow_html=True)
    with c3:
        st.markdown(ui.card("Best price", f'{best["best_odds"]:.2f}',
                            f'{best["best_book"]} · avg {best["avg_odds"]:.2f}', ui.BLUE),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(ui.card("Confidence", f'{best["confidence"]*100:.0f}%',
                            f'{data["book_count"]} books · {data["market_source"]}',
                            ui.BLUE if best["confidence"] >= 0.7 else ui.YELLOW),
                    unsafe_allow_html=True)
    with c5:
        risk_color = {"Low": ui.GREEN, "Medium": ui.YELLOW, "High": ui.RED}[best["risk"]]
        st.markdown(ui.card("Risk level", best["risk"],
                            f'¼-Kelly stake {best["kelly_stake"]*100:.1f}% of bankroll', risk_color),
                    unsafe_allow_html=True)

    st.markdown('<div class="section-head">All selections priced</div>', unsafe_allow_html=True)
    df = selections_frame(sels)
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Value %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Edge (pts)": st.column_config.NumberColumn(format="%+.1f"),
            "True %": st.column_config.NumberColumn(format="%.1f%%"),
            "Book %": st.column_config.NumberColumn(format="%.1f%%"),
            "Best odds": st.column_config.NumberColumn(format="%.2f"),
            "Conf.": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
        },
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="section-head">True probability vs bookmaker probability</div>',
                    unsafe_allow_html=True)
        ui.render(ui.probability_chart(sels), key="probchart")
        st.markdown('<div class="section-head">Value by selection</div>', unsafe_allow_html=True)
        ui.render(ui.value_chart(sels), key="valuechart")
    with right:
        st.markdown('<div class="section-head">Value gauge</div>', unsafe_allow_html=True)
        ui.render(ui.gauge(best["value_pct"], best["signal"]), key="gauge")
        st.markdown('<div class="section-head">Signal contributions (pick side)</div>',
                    unsafe_allow_html=True)
        ui.render(ui.components_chart(a.components), key="comps")

    st.markdown('<div class="section-head">Why this bet?</div>', unsafe_allow_html=True)
    st.markdown(ui.reasoning_block(best["reasoning"]), unsafe_allow_html=True)

    with st.expander("Full reasoning for every selection"):
        for s in sels:
            st.markdown(f"**{s['selection']}** — {s['signal']} ({s['value_pct']:+.1f}%)")
            st.markdown(ui.reasoning_block(s["reasoning"]), unsafe_allow_html=True)
            st.write("")

    st.markdown('<div class="section-head">Bookmaker board</div>', unsafe_allow_html=True)
    keys = {"A": "Side A", "B": "Side B"}
    if a.sport == "nba":
        keys = {"A": "Home ML", "B": "Away ML"}
    pick_key = st.radio("Show odds for", list(keys), horizontal=True,
                        format_func=lambda k: keys[k], key="odds_key")
    ui.render(ui.odds_distribution(a.market["moneyline"], pick_key, keys[pick_key]), key="oddsdist")

    if a.sport == "nba" and data.get("totals"):
        t = data["totals"]
        st.markdown('<div class="section-head">Over / Under projection</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(ui.card("Model total", f'{t["expected_total"]:.1f}',
                                f'line {t["line"]:.1f}', "#e8eef9"), unsafe_allow_html=True)
        with c2:
            st.markdown(ui.card("Edge vs line", f'{t["edge_points"]:+.1f} pts',
                                f'sigma {t["sigma"]:.1f}', ui.GREEN if t["edge_points"] > 0 else ui.RED),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(ui.card("P(Over)", fmt_pct(t["p_over"]),
                                f'P(Under) {fmt_pct(t["p_under"])}', ui.BLUE), unsafe_allow_html=True)
        with c4:
            st.markdown(ui.card("Pace / Ratings", f'{t["pace"]:.1f}',
                                f'OFF {t["combined_off"]:.1f} · DEF {t["combined_def"]:.1f}',
                                ui.VIOLET), unsafe_allow_html=True)

    with st.expander("Data quality & sources"):
        st.json(data["data_quality"])
        st.caption(f"Market board: {data['market_note']} · fetched {a.market.get('fetched_at')}")

    st.markdown(
        f'<div class="ss-foot">Analysis generated {data["created_at"]} · market source '
        f'<b>{data["market_source"]}</b> · {data["book_count"]} bookmakers · '
        f'event {data["event_time"] or "not set"}. Value = true probability x best decimal odds - 1. '
        f'This is a statistical model, not financial advice.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <div class="ss-logo">S</div>
          <div>
            <div style="font-weight:800;font-size:17px;line-height:1.1;">SureStats</div>
            <div style="color:#8b9ab5;font-size:11.5px;letter-spacing:.06em;">TENNIS &amp; NBA</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="ss-card-label">Data connections</div>', unsafe_allow_html=True)
    for name, ok, env in api_status():
        cls = "green" if ok else "yellow"
        label = "live" if ok else "fallback"
        st.markdown(f'{ui.pill(f"{name}: {label}", cls)}', unsafe_allow_html=True)
        if not ok:
            st.caption(f"set `{env}` in .env to go live")

    st.markdown("---")
    st.markdown('<div class="ss-card-label">Engine</div>', unsafe_allow_html=True)
    st.caption(
        f"ELO {WEIGHTS['elo']:.0%} · Form {WEIGHTS['form']:.0%} · "
        f"Surface/Home {WEIGHTS['surface_or_home']:.0%} · H2H {WEIGHTS['h2h']:.0%} · "
        f"Fatigue/Injury {WEIGHTS['fatigue_injury']:.0%}"
    )
    st.caption(f"GREEN ≥ {settings.value_green:.0f}% · YELLOW ≥ {settings.value_yellow:.0f}%")
    st.caption(f"Odds refresh: every {settings.odds_refresh_seconds // 60} min")

    st.markdown("---")
    stats = db.analysis_stats()
    st.markdown('<div class="ss-card-label">Session database</div>', unsafe_allow_html=True)
    st.caption(f"{int(stats.get('n', 0))} selections logged · "
               f"{int(stats.get('greens', 0))} green signals")
    if st.button("Clear analysis history", width="stretch"):
        db.delete_all_analyses()
        st.success("History cleared")

# ---------------------------------------------------------------------------
# hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-top">
        <div class="ss-logo">S</div>
        <div>
          <div class="ss-title">SureStats Bot</div>
          <div class="ss-sub">Tennis &amp; NBA value detection · ELO + calibrated probability + best-price pricing</div>
        </div>
      </div>
      <h1>Find the price the market got wrong.</h1>
      <p>Enter a matchup. The engine builds both sides' profiles, blends five weighted signals
      (ELO, form, surface/home, head-to-head, fatigue &amp; injury), calibrates the result with
      scikit-learn, then compares true probability against the best of 24 bookmakers to compute
      value. NBA fixtures also get a pace-driven Over/Under projection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# input panel
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="ss-card-label">Match setup</div>', unsafe_allow_html=True)
    r1 = st.columns([1, 2.2, 1.1, 1.1])
    with r1[0]:
        sport_label = st.selectbox("Sport", ["Tennis", "NBA"], index=0)
    sport = sport_label.lower()

    presets = TENNIS_PRESETS if sport == "tennis" else NBA_PRESETS
    with r1[1]:
        quick = st.selectbox("Matchup", ["Custom matchup"] + presets, index=1)
        if quick == "Custom matchup":
            match = st.text_input(
                "Type the match",
                placeholder="Djokovic vs Alcaraz" if sport == "tennis" else "Lakers vs Warriors",
            )
        else:
            match = quick
    with r1[2]:
        event_date = st.date_input("Date", value=date.today() + timedelta(days=1))
    with r1[3]:
        event_time = st.time_input("Start time (UTC)", value=time(18, 0))

    r2 = st.columns([1.1, 1.1, 1.4, 1.2])
    surface = "hard"
    home_team = None
    if sport == "tennis":
        with r2[0]:
            surface = st.selectbox("Surface", ["hard", "clay", "grass"], index=0)
        with r2[1]:
            st.caption("Players available")
            st.caption(f"{len(tennis_player_names())} in the reference DB")
    else:
        with r2[0]:
            st.caption("Home court")
            st.caption("First team is home by default")
        with r2[1]:
            st.caption("Teams available")
            st.caption(f"{len(nba_team_names())} in the reference DB")
    with r2[2]:
        auto_refresh = st.toggle("Auto-refresh odds", value=True,
                                 help="Re-prices the board on a timer (default 10 min)")
    with r2[3]:
        analyze = st.button("Analyze", width="stretch", type="primary")

if analyze:
    if not match or " vs " not in match.lower().replace(" v ", " vs "):
        st.error("Enter the match as `Team A vs Team B` (e.g. Djokovic vs Alcaraz).")
    else:
        st.session_state["params"] = {
            "sport": sport, "match": match, "surface": surface,
            "event_time": f"{event_date.isoformat()}T{event_time.strftime('%H:%M')}Z",
        }
        cache_clear("odds:")
        log.info("analyze requested: %s", st.session_state["params"])

# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
tab_analyze, tab_model, tab_history, tab_system = st.tabs(
    ["Analyzer", "Model & Accuracy", "History", "System"]
)

with tab_analyze:
    params = st.session_state.get("params")
    if not params:
        render_empty()
    else:
        run_every = settings.odds_refresh_seconds if auto_refresh else None

        @st.fragment(run_every=run_every)
        def live_results() -> None:
            p = st.session_state.get("params")
            if not p:
                return
            try:
                with st.spinner("Pulling profiles, pricing the board and calibrating..."):
                    analysis = engine.analyze(p["sport"], p["match"], p["event_time"],
                                              surface=p["surface"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")
                log.exception("analysis failed")
                return
            st.session_state["analysis"] = analysis.to_dict()
            render_analysis(analysis)
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            c1, c2 = st.columns([4, 1])
            with c1:
                st.caption(f"Board refreshed at {stamp}. "
                           + ("Auto-refresh is on." if auto_refresh else "Auto-refresh is off."))
            with c2:
                if st.button("Refresh now", width="stretch"):
                    cache_clear("odds:")
                    st.rerun(scope="fragment")

        live_results()

# ---------------------------------------------------------------------------
# model tab
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(sport: str) -> dict:
    return backtest.run_and_store(sport)


with tab_model:
    st.markdown('<div class="section-head">Historical accuracy & strategy performance</div>',
                unsafe_allow_html=True)
    bt_sport = st.radio("Model", ["tennis", "nba"], horizontal=True,
                        format_func=lambda s: "Tennis" if s == "tennis" else "NBA",
                        key="bt_sport")
    if st.button("Run fresh backtest", key="run_bt"):
        cached_backtest.clear()
        st.rerun()

    bt = cached_backtest(bt_sport)
    detail = bt.get("detail", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(ui.card("Accuracy", fmt_pct(bt["accuracy"]),
                            f'{bt["n_matches"]} historical matches', ui.BLUE), unsafe_allow_html=True)
    with c2:
        st.markdown(ui.card("Brier score", f'{bt["brier"]:.4f}',
                            "lower is better", "#e8eef9"), unsafe_allow_html=True)
    with c3:
        st.markdown(ui.card("Log loss", f'{bt["log_loss"]:.4f}',
                            "calibration quality", "#e8eef9"), unsafe_allow_html=True)
    with c4:
        st.markdown(ui.card("Value bets", f'{bt["n_bets"]}',
                            f'avg value {detail.get("avg_value_pct", 0):.1f}%', ui.VIOLET),
                    unsafe_allow_html=True)
    with c5:
        roi_color = ui.GREEN if bt["roi_pct"] > 0 else ui.RED
        st.markdown(ui.card("Simulated ROI", f'{bt["roi_pct"]:+.1f}%',
                            f'hit rate {detail.get("hit_rate_bets", 0)*100:.1f}%', roi_color),
                    unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-head">Calibration: predicted vs actual</div>',
                    unsafe_allow_html=True)
        ui.render(ui.calibration_chart(detail.get("calibration", [])), key="calib")
    with right:
        st.markdown('<div class="section-head">Cumulative profit (value strategy)</div>',
                    unsafe_allow_html=True)
        ui.render(ui.profit_curve(detail.get("profit_curve", [])), key="profit")

    st.markdown('<div class="section-head">How the probability is built</div>',
                unsafe_allow_html=True)
    wdf = pd.DataFrame(
        [{"Signal": k.replace("_", " ").title(), "Weight": f"{v:.0%}"} for k, v in WEIGHTS.items()]
    )
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(wdf, hide_index=True, width="stretch")
        cal = elo_mod.calibration_summary(bt_sport)
        st.caption(f"Platt calibration: logit_out = {cal['coef']:.3f} x logit_blend "
                   f"{cal['intercept']:+.3f} (scikit-learn LogisticRegression)")
    with c2:
        st.markdown(
            """
            <div class="ss-card">
              <div class="ss-card-label">Pipeline</div>
              <div class="ss-reason"><span class="n">01</span><span>Each signal is converted to its own
              probability (ELO via the logistic curve, form via weighted last-10, surface/home via
              surface ELO or net rating, H2H via Laplace-smoothed win rate, fatigue/injury via a
              bounded penalty).</span></div>
              <div class="ss-reason"><span class="n">02</span><span>The five probabilities are blended
              with the weights on the left, then passed through a Platt calibrator fitted on the
              historical results.</span></div>
              <div class="ss-reason"><span class="n">03</span><span>Every bookmaker price is collected;
              the market probability is the vig-free average across books, and the best price is the
              maximum decimal odds.</span></div>
              <div class="ss-reason"><span class="n">04</span><span>Value % = true probability x best
              odds - 1. ≥8% is GREEN, 3-8% YELLOW, below 0% RED.</span></div>
              <div class="ss-reason"><span class="n">05</span><span>Confidence reflects data quality
              (book count, live vs fallback sources, form sample) - never the size of the edge.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# history tab
# ---------------------------------------------------------------------------
with tab_history:
    st.markdown('<div class="section-head">Analysis history (SQLite)</div>', unsafe_allow_html=True)
    sport_filter = st.radio("Filter", ["all", "tennis", "nba"], horizontal=True, key="hist_filter")
    rows = db.recent_analyses(300, None if sport_filter == "all" else sport_filter)
    if not rows:
        st.info("No analyses stored yet. Run one from the Analyzer tab.")
    else:
        ui.render(ui.history_chart(rows), key="histchart")
        hdf = pd.DataFrame([
            {
                "When (UTC)": str(r["created_at"]).replace("T", " "),
                "Sport": r["sport"].upper(),
                "Match": r["match"],
                "Selection": r["selection"],
                "Market": r["market"],
                "True %": round((r["true_prob"] or 0) * 100, 1),
                "Book %": round((r["book_prob"] or 0) * 100, 1),
                "Best odds": round(r["best_odds"] or 0, 2),
                "Book": r["best_book"],
                "Value %": round(r["value_pct"] or 0, 1),
                "Signal": r["signal"],
                "Conf %": round((r["confidence"] or 0) * 100),
                "Risk": r["risk"],
            }
            for r in rows
        ])
        st.dataframe(hdf, hide_index=True, width="stretch", height=420)
        st.download_button(
            "Download CSV", hdf.to_csv(index=False).encode("utf-8"),
            file_name=f"surestats_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# system tab
# ---------------------------------------------------------------------------
with tab_system:
    st.markdown('<div class="section-head">Runtime & configuration</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="ss-card">
              <div class="ss-card-label">Environment</div>
              <div class="ss-reason"><span class="n">›</span><span>Database:
              <span class="mono">{settings.db_path}</span></span></div>
              <div class="ss-reason"><span class="n">›</span><span>Log file:
              <span class="mono">{settings.log_dir / 'surestats.log'}</span></span></div>
              <div class="ss-reason"><span class="n">›</span><span>Odds refresh:
              {settings.odds_refresh_seconds}s · cache TTL {settings.cache_ttl_seconds}s</span></div>
              <div class="ss-reason"><span class="n">›</span><span>Demo mode:
              {settings.demo_mode} · live data {settings.live_data}</span></div>
              <div class="ss-reason"><span class="n">›</span><span>ELO K tennis
              {settings.elo_k_tennis:.0f} · K NBA {settings.elo_k_nba:.0f} · home adv
              {settings.home_advantage_nba:.0f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        rows = "".join(
            f'<div class="ss-reason"><span class="n">›</span><span>{name}: '
            f'<b style="color:{ui.GREEN if ok else ui.YELLOW}">'
            f'{"connected" if ok else "not configured (fallback active)"}</b> '
            f'<span class="mono" style="color:#8b9ab5">{env}</span></span></div>'
            for name, ok, env in api_status()
        )
        st.markdown(
            f'<div class="ss-card"><div class="ss-card-label">API keys</div>{rows}'
            f'<div class="ss-card-sub">Add keys to <span class="mono">.env</span> and restart to '
            f'switch every provider to live data. The engine works fully without them.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-head">Recent log lines</div>', unsafe_allow_html=True)
    log_path = settings.log_dir / "surestats.log"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
        st.code("\n".join(lines), language="log")
    else:
        st.caption("No log file yet.")

    st.markdown('<div class="section-head">Stored backtests</div>', unsafe_allow_html=True)
    for s in ("tennis", "nba"):
        b = db.latest_backtest(s)
        if b:
            st.caption(
                f"{s.upper()}: n_bets={b['n_bets']} · accuracy={b['accuracy']:.3f} · "
                f"brier={b['brier']:.4f} · log loss={b['log_loss']:.4f} · ROI={b['roi_pct']:+.1f}% "
                f"· {b['created_at']}"
            )

st.markdown(
    '<div class="ss-foot">SureStats Bot v1.0 · Built for research and entertainment. '
    'Value betting carries risk of loss; stake only what you can afford to lose and check your '
    'local regulations.</div>',
    unsafe_allow_html=True,
)
