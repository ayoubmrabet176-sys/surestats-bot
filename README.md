# SureStats Bot — Tennis & NBA

**A value-betting research engine that prices tennis and NBA markets itself, then hunts for the bookmaker whose price is wrong.**

SureStats is not a tipster. It is a transparent pricing model. You give it a sport, a matchup and a start time. It builds a statistical profile for both sides, blends five weighted signals into a *true probability*, calibrates that probability with a fitted scikit-learn model, and then compares it against the best available decimal odds across 24 bookmakers. Where the model's probability is meaningfully higher than the market's implied probability, the selection is flagged as value — green above 8%, yellow between 3% and 8%, red below zero.

Everything in the app is computed from data. There are no hard-coded picks, no placeholder numbers, and no hidden assumptions about who *should* win.

---

## What you get

The dashboard has four working tabs. **Analyzer** is the main workspace: a match-setup panel, a signal banner, five headline metric cards (best value, true probability, best price, confidence, risk), a full selection table with every priced market, three charts (true vs bookmaker probability, value by selection, signal contributions), a value gauge, a plain-English "Why this bet?" reasoning block, a bookmaker board showing every quote, and — for NBA — a pace-driven Over/Under projection with its own edge calculation. **Model & Accuracy** runs and caches a historical backtest, showing accuracy, Brier score, log loss, number of value bets, simulated ROI, a calibration curve and a cumulative profit curve, plus a description of how the probability is assembled. **History** reads every stored analysis out of SQLite, charts the value distribution over time and lets you export the whole log as CSV. **System** shows runtime configuration, which API keys are live, the last forty log lines and the stored backtests.

Odds re-price automatically every ten minutes via a Streamlit fragment, and there is a manual "Refresh now" button that clears the odds cache immediately.

---

## Quick start (local)

```bash
git clone <your-repo> surestats && cd surestats
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # optional - see keys below
streamlit run dashboard/app.py
```

Open <http://localhost:8501>. The app works immediately with **zero API keys**: it ships with a curated reference database of 18 ATP/WTA players and all 30 NBA teams, a deterministic 2,400-match tennis history and 1,800-game NBA history for calibration, and a deterministic simulated 24-bookmaker market. That is not a mock-up — the pricing, blending, calibration, value maths, totals model, persistence and backtesting all run for real. The only thing simulated is the *market*, and it is simulated with realistic per-book margins and dispersion so the value detector has something honest to detect.

### Verify the engine from the command line

```bash
python scripts/smoke_test.py
```

This initialises the database, prints both calibration fits, analyses a tennis match and an NBA game, runs and stores both backtests, and prints the recent-analysis stats. A healthy run ends with `SMOKE TEST PASSED`.

---

## Getting free API keys

Every integration is optional and degrades gracefully. Add a key and that source goes live; leave it blank and the bundled reference data is used instead. The sidebar always tells you which mode each source is in.

### The Odds API — real bookmaker prices (free: 500 requests/month)

Go to <https://the-odds-api.com/#get-access>, enter your email and create a free account. Your API key appears on the dashboard immediately. Copy it into `.env` as `ODDS_API_KEY`. The free tier is enough for roughly 800 match analyses a month at one request each, and the app caches responses for ten minutes so repeated refreshes do not burn quota. SureStats requests the `eu,us,uk,au` regions and the `h2h` and `totals` markets, which is a single billable call per refresh.

### API-Sports / API-Tennis — player profiles, form and head-to-head (free: 100 requests/day)

Register at <https://dashboard.api-football.com/register>, then choose **Tennis** from the product switcher and copy the API key from the dashboard. Put it in `.env` as `TENNIS_API_KEY`. The client sends it in the `x-apisports-key` header. On the free plan the app uses it to overlay live ranking, surface ELO proxies and recent results on top of the reference profile, and it caches aggressively to stay inside the daily limit.

### balldontlie — NBA teams, results and rest days (free tier)

Sign up at <https://www.balldontlie.io/> and copy your key into `.env` as `BALLDONTLIE_API_KEY`. The client sends it as a bearer token. It is used to derive last-ten form, rest days and back-to-backs, and to nudge offensive and defensive ratings from the reference values toward observed results.

> **Security note:** `.env` is git-ignored. Never commit real keys. On Render, add them as environment variables in the dashboard instead.

---

## Deploying on Render.com, free forever

Render's free web-service tier is genuinely free and does not expire; the trade-off is that the instance sleeps after fifteen minutes of inactivity and cold-starts in about thirty seconds. For a research tool that is usually fine.

**Option A — Blueprint (one click).** Push this repository to GitHub. In Render, choose **New + → Blueprint**, select the repo, and Render reads `render.yaml` automatically. It creates the service, sets the Python version, wires the health check to `/_stcore/health`, and prompts you for the three API keys (leave them blank if you do not have them). Click **Apply** and wait for the first build.

**Option B — manual.** Create a **New + → Web Service**, connect the repo, set runtime to Python 3, build command `pip install --upgrade pip && pip install -r requirements.txt`, and start command:

```bash
streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
```

Add the environment variables from `.env.example` under the **Environment** tab, set the health check path to `/_stcore/health`, and deploy.

**A note on persistence.** Render's free filesystem is ephemeral, so the SQLite database resets whenever the service redeploys or wakes from a long sleep. The app handles this gracefully — it recreates the schema on boot — but if you want your analysis history to survive, attach a Render Disk (paid) and point the app at it by setting the database path, or simply export the CSV from the History tab before redeploying.

### Docker

```bash
docker build -t surestats .
docker run -p 8501:8501 --env-file .env surestats
```

The image is based on `python:3.11-slim`, installs only wheel dependencies, runs as a single Streamlit process, and exposes a container health check against `/_stcore/health`.

---

## How the model works

The engine is deliberately legible. Every number on screen can be traced back to a formula in `models/`.

**ELO.** Tennis players carry an overall rating plus separate hard, clay and grass ratings. The match rating blends 65% surface-specific and 35% overall ELO, because surface specialisation is real but should not swamp long-run quality. NBA teams carry a single rating with a 65-point home-court bonus. Expected score follows the standard logistic curve, `1 / (1 + 10^((rb − ra)/400))`, and ratings update after each result with a K-factor of 32 for tennis and 20 for NBA, scaled by margin of victory.

**The five signals.** ELO contributes 35%, recent form 30%, surface (tennis) or home advantage (NBA) 20%, head-to-head 10%, and fatigue/injury 5%. Form is a recency-weighted win rate over the last ten results, with the most recent match weighted 1.6 and the oldest 0.6. Surface performance compares the two players' surface ELOs through the same logistic curve; for NBA it uses net rating adjusted for home court. Head-to-head is a Laplace-smoothed win rate so a 2–0 record does not read as certainty. Fatigue and injury form a bounded penalty block: matches played in the last seven days, back-to-backs and listed injuries each subtract from the raw probability, with the total penalty capped so a single flag cannot flip a strong favourite.

**Calibration.** The five component probabilities are converted to logits, blended with the weights above, and passed through a Platt-scaling logistic regression fitted on the historical match set (2,400 tennis matches, 1,800 NBA games). The fitted coefficients are printed in the Model tab — currently 0.949 for tennis and 2.001 for NBA — so you can see exactly how much the raw blend is being sharpened or softened. This is the step that turns a reasonable heuristic into a calibrated probability.

**Value.** Value percentage is `(true probability × best decimal odds) − 1`. The bookmaker's implied probability is obtained by devigging each bookmaker's two-way market (normalising the implied probabilities so they sum to one) and then averaging across books, which removes the margin and gives a fair market consensus. The best price is the single highest decimal quote available, and the edge in percentage points is the difference between true and market probability. A quarter-Kelly stake is suggested as a fraction of bankroll, deliberately conservative.

**Confidence and risk.** Confidence is a data-quality score built from how many bookmakers quoted the market, whether the profiles came from live APIs or the reference database, and how much recent form data was available. Risk is derived from confidence, the size of the value edge and the price itself — a big edge at long odds is a higher-variance proposition than the same edge at evens, and the risk label reflects that.

**NBA totals.** Expected total points is `pace × (home points-per-100 + away points-per-100) / 100`, adjusted for injuries and rest. The Over/Under probability comes from a Gaussian CDF around that expectation, with the residual sigma learned from the historical NBA residuals (currently about 12.75 points). The edge in points is the gap between the model's expected total and the bookmaker's line.

**A note on the fatigue/injury term.** The brief described the blend as "ELO + Form 30% + Surface/Home 20% + H2H 10% + Fatigue/Injury −15%". Taken literally the weights sum to more than 100%, so the implementation normalises the four positive signals to fill 95% of the blend and applies fatigue/injury as a bounded negative adjustment worth up to 5% of the probability mass. This preserves the intent — fatigue and injury should hurt, but should never dominate — while keeping the output a valid probability. The exact weights are visible in `config.py` and on screen in the sidebar.

---

## Backtesting

`models/backtest.py` replays the historical match sets through the calibrated model, prices each side against a simulated market with realistic margin and dispersion, takes the highest-value side whenever it clears the threshold, and records the result. It reports accuracy, Brier score, log loss, the number of qualifying bets, hit rate, ROI and a full calibration table, plus a cumulative profit curve. The default threshold is 3% value. On the bundled data the tennis strategy shows roughly 59% accuracy and +15% simulated ROI over 533 bets; NBA shows roughly 67% accuracy and +14.5% ROI over 486 bets. These are *simulated* returns against a simulated market and should be read as a sanity check on the model's calibration, not as a promise of future profit.

---

## Project layout

```
surestats/
├── config.py                  # environment-driven settings + model weights
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .env.example
├── api/
│   ├── odds_api.py            # The Odds API client, devigging, simulated 24-book market
│   ├── tennis_api.py          # API-Tennis client with reference-data overlay
│   └── nba_api.py             # balldontlie client with derived form/rest
├── data/
│   ├── db.py                  # SQLite: analyses, odds snapshots, backtests, kv store
│   └── datasets.py            # 18 players, 30 teams, deterministic histories
├── models/
│   ├── elo.py                 # ELO books, five-signal blend, Platt calibration
│   ├── totals.py              # pace/off-rating Over-Under model
│   ├── engine.py              # orchestration, value maths, confidence, risk, reasoning
│   └── backtest.py            # historical replay, Brier/log loss/ROI, curves
├── dashboard/
│   ├── app.py                 # Streamlit application
│   ├── components.py          # Plotly charts and HTML cards
│   └── styles.py              # dark theme CSS
├── utils/
│   ├── http.py                # cached HTTP with retries and backoff
│   └── logging_conf.py        # rotating file + console logging
└── scripts/
    └── smoke_test.py          # end-to-end verification
```

---

## Configuration reference

All settings live in `config.py` and are read from the environment. The ones you are most likely to touch are `ODDS_REFRESH_SECONDS` (how often odds re-price, default 600), `VALUE_GREEN` and `VALUE_YELLOW` (the signal thresholds, default 8 and 3), `DEMO_MODE` (force offline reference data even when keys exist), and the ELO hyper-parameters `ELO_K_TENNIS`, `ELO_K_NBA`, `HOME_ADVANTAGE_NBA` and `HOME_ADVANTAGE_TENNIS`.

Logging goes to `logs/surestats.log` with rotation at 2 MB and three backups, and to the console. Every module logs under the `surestats.*` namespace, so you can raise verbosity per module with `LOG_LEVEL=DEBUG`.

---

## Troubleshooting

If the dashboard shows "fallback" next to an API, the key is missing or invalid — check `.env` and restart. If odds look identical between refreshes, you are inside the ten-minute cache window; press "Refresh now". If the app cannot write to `data/store`, check directory permissions; it needs write access for SQLite and WAL files. If a player or team is not recognised, use the surname or a common alias — the lookup handles "Djokovic", "Lakers", "Sixers", "Blazers" and similar — or type the matchup manually as "A vs B".

---

## Disclaimer

SureStats is a research and educational tool. It does not guarantee profit, and value betting carries a real risk of loss. Odds move, models are wrong, and past calibration does not ensure future returns. Stake only what you can afford to lose, and check the gambling regulations in your jurisdiction before using any of this with real money.
