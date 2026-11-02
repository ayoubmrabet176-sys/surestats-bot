"""Custom CSS for the SureStats dashboard (dark, editorial, high-contrast)."""
from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #070b14;
  --panel: #0e1524;
  --panel-2: #121b2e;
  --line: #1e2a42;
  --text: #e8eef9;
  --muted: #8b9ab5;
  --green: #00d68f;
  --yellow: #ffb020;
  --red: #ff4d6d;
  --blue: #4d8dff;
  --violet: #9d7bff;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(77,141,255,0.16), transparent 60%),
    radial-gradient(900px 500px at 92% 0%, rgba(0,214,143,0.12), transparent 55%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #080d17 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }

h1, h2, h3, h4 { letter-spacing: -0.02em; }
p, li, span, label { color: var(--text); }

/* ---------- hero ---------- */
.ss-hero {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 30px 34px;
  background:
    linear-gradient(135deg, rgba(77,141,255,0.14), rgba(0,214,143,0.07) 45%, rgba(157,123,255,0.10)),
    var(--panel);
  overflow: hidden;
  margin-bottom: 22px;
}
.ss-hero:before {
  content: "";
  position: absolute; inset: 0;
  background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: radial-gradient(600px 200px at 20% 0%, #000 30%, transparent 75%);
}
.ss-hero-top { display: flex; align-items: center; gap: 12px; }
.ss-logo {
  width: 42px; height: 42px; border-radius: 12px;
  background: linear-gradient(135deg, var(--green), var(--blue));
  display: grid; place-items: center; font-weight: 800; font-size: 19px; color: #04121a;
  box-shadow: 0 8px 24px rgba(0,214,143,0.28);
}
.ss-title { font-size: 30px; font-weight: 800; margin: 0; line-height: 1.1; }
.ss-sub { color: var(--muted); font-size: 14px; margin-top: 4px; }
.ss-hero h1 { font-size: 34px; margin: 18px 0 6px; }
.ss-hero p { color: var(--muted); max-width: 760px; margin: 0; font-size: 14.5px; line-height: 1.6; }

/* ---------- cards ---------- */
.ss-card {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 20px;
  height: 100%;
}
.ss-card-label {
  font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; margin-bottom: 8px;
}
.ss-card-value { font-size: 30px; font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; }
.ss-card-sub { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* ---------- signal banner ---------- */
.ss-banner {
  border-radius: 18px; padding: 22px 26px; margin: 6px 0 18px;
  border: 1px solid var(--line); position: relative; overflow: hidden;
}
.ss-banner .tag { font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; opacity: .85; }
.ss-banner .head { font-size: 24px; font-weight: 800; margin-top: 6px; letter-spacing: -0.02em; }
.ss-banner .meta { color: rgba(255,255,255,0.78); font-size: 13.5px; margin-top: 6px; }
.ss-green  { background: linear-gradient(120deg, rgba(0,214,143,0.22), rgba(0,214,143,0.05)); border-color: rgba(0,214,143,0.45); }
.ss-yellow { background: linear-gradient(120deg, rgba(255,176,32,0.20), rgba(255,176,32,0.04)); border-color: rgba(255,176,32,0.42); }
.ss-red    { background: linear-gradient(120deg, rgba(255,77,109,0.18), rgba(255,77,109,0.04)); border-color: rgba(255,77,109,0.38); }

/* ---------- badges / pills ---------- */
.ss-pill {
  display: inline-block; padding: 4px 11px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid var(--line); background: rgba(255,255,255,0.04); color: var(--text);
}
.ss-pill.green  { color: var(--green);  border-color: rgba(0,214,143,0.45);  background: rgba(0,214,143,0.10); }
.ss-pill.yellow { color: var(--yellow); border-color: rgba(255,176,32,0.45); background: rgba(255,176,32,0.10); }
.ss-pill.red    { color: var(--red);    border-color: rgba(255,77,109,0.45); background: rgba(255,77,109,0.10); }
.ss-pill.blue   { color: var(--blue);   border-color: rgba(77,141,255,0.45); background: rgba(77,141,255,0.10); }

/* ---------- reasoning ---------- */
.ss-reason { display: flex; gap: 12px; padding: 9px 0; border-bottom: 1px dashed rgba(255,255,255,0.06); font-size: 14px; line-height: 1.55; }
.ss-reason:last-child { border-bottom: none; }
.ss-reason .n { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 12px; padding-top: 2px; min-width: 20px; }

/* ---------- streamlit widgets ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stTextInput input, .stDateInput input {
  background: var(--panel-2) !important; border-color: var(--line) !important; color: var(--text) !important;
}
.stTextInput input, .stNumberInput input { color: var(--text) !important; }
.stButton > button {
  background: linear-gradient(135deg, var(--green), #00b4c8);
  color: #04121a; font-weight: 700; border: none; border-radius: 12px;
  padding: 0.62rem 1.4rem; letter-spacing: 0.01em;
  box-shadow: 0 10px 26px rgba(0,214,143,0.22);
}
.stButton > button:hover { filter: brightness(1.06); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: 10px 10px 0 0; padding: 8px 16px;
  color: var(--muted); font-weight: 600;
}
.stTabs [aria-selected="true"] { color: var(--text) !important; background: rgba(255,255,255,0.05); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stMetricValue"] { color: var(--text); }
hr { border-color: var(--line); }
details { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }
.section-head { font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); font-weight: 700; margin: 26px 0 12px; }
.ss-foot { color: var(--muted); font-size: 12px; margin-top: 26px; line-height: 1.6; }
</style>
"""
