"""Reusable HTML cards and Plotly charts for the SureStats dashboard."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

GREEN, YELLOW, RED, BLUE, VIOLET = "#00d68f", "#ffb020", "#ff4d6d", "#4d8dff", "#9d7bff"
MUTED = "#8b9ab5"
GRID = "rgba(255,255,255,0.06)"


def _base_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e8eef9", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#121b2e", bordercolor="#1e2a42", font_color="#e8eef9"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def signal_class(signal: str) -> str:
    if "GREEN" in signal:
        return "green"
    if "YELLOW" in signal:
        return "yellow"
    return "red"


def signal_color(signal: str) -> str:
    return {"green": GREEN, "yellow": YELLOW, "red": RED}[signal_class(signal)]


def card(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    style = f"color:{color};" if color else ""
    return f"""
    <div class="ss-card">
      <div class="ss-card-label">{label}</div>
      <div class="ss-card-value" style="{style}">{value}</div>
      <div class="ss-card-sub">{sub}</div>
    </div>
    """


def banner(signal: str, headline: str, meta: str) -> str:
    cls = signal_class(signal)
    return f"""
    <div class="ss-banner ss-{cls}">
      <div class="tag">{signal}</div>
      <div class="head">{headline}</div>
      <div class="meta">{meta}</div>
    </div>
    """


def pill(text: str, cls: str = "") -> str:
    return f'<span class="ss-pill {cls}">{text}</span>'


def reasoning_block(items: list[str]) -> str:
    rows = "".join(
        f'<div class="ss-reason"><span class="n">{i+1:02d}</span><span>{t}</span></div>'
        for i, t in enumerate(items)
    )
    return f'<div class="ss-card">{rows}</div>'


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def probability_chart(selections: list[dict[str, Any]]) -> go.Figure:
    names = [s["selection"] for s in selections]
    fig = go.Figure()
    fig.add_bar(name="True probability", x=names, y=[s["true_prob"] * 100 for s in selections],
                marker_color=GREEN, text=[f"{s['true_prob']*100:.1f}%" for s in selections],
                textposition="outside")
    fig.add_bar(name="Bookmaker probability", x=names, y=[s["book_prob"] * 100 for s in selections],
                marker_color=BLUE, text=[f"{s['book_prob']*100:.1f}%" for s in selections],
                textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="Probability (%)", yaxis_range=[0, 110])
    return _base_layout(fig, 320)


def value_chart(selections: list[dict[str, Any]]) -> go.Figure:
    names = [s["selection"] for s in selections]
    values = [s["value_pct"] for s in selections]
    colors = [signal_color(s["signal"]) for s in selections]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:+.1f}%" for v in values], textposition="outside"))
    fig.add_vline(x=8, line_dash="dot", line_color=GREEN, annotation_text="GREEN 8%")
    fig.add_vline(x=3, line_dash="dot", line_color=YELLOW, annotation_text="YELLOW 3%")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)")
    fig.update_layout(xaxis_title="Value % (true prob x best odds - 1)", showlegend=False)
    return _base_layout(fig, 300)


def components_chart(components: dict[str, float]) -> go.Figure:
    labels = {"elo": "ELO", "form": "Form (last 10)", "surface_or_home": "Surface / Home",
              "h2h": "Head-to-head", "fatigue_injury": "Fatigue / Injury"}
    keys = list(labels)
    vals = [components.get(k, 0.5) * 100 for k in keys]
    colors = [GREEN if v >= 50 else RED for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=[labels[k] for k in keys], orientation="h",
                           marker_color=colors, text=[f"{v:.1f}%" for v in vals],
                           textposition="outside"))
    fig.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                  annotation_text="coin flip")
    fig.update_layout(xaxis_range=[0, 100], xaxis_title="Probability the selection wins this signal",
                      showlegend=False)
    return _base_layout(fig, 300)


def odds_distribution(quotes: list[dict[str, Any]], sel_key: str, label: str) -> go.Figure:
    rows = sorted([q for q in quotes if q.get("selection") == sel_key], key=lambda q: q["price"])
    books = [q["bookmaker"] for q in rows]
    prices = [q["price"] for q in rows]
    best = max(prices) if prices else 0
    colors = [GREEN if p == best else BLUE for p in prices]
    fig = go.Figure(go.Bar(x=books, y=prices, marker_color=colors,
                           text=[f"{p:.2f}" for p in prices], textposition="outside"))
    fig.update_layout(xaxis_title=f"{label} - decimal odds by bookmaker",
                      yaxis_title="Decimal odds", showlegend=False,
                      yaxis_range=[min(prices) * 0.97 if prices else 0, best * 1.06])
    fig.update_xaxes(tickangle=-45)
    return _base_layout(fig, 360)


def gauge(value_pct: float, signal: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [-15, 30], "tickcolor": MUTED},
            "bar": {"color": signal_color(signal), "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [-15, 0], "color": "rgba(255,77,109,0.16)"},
                {"range": [0, 3], "color": "rgba(255,255,255,0.05)"},
                {"range": [3, 8], "color": "rgba(255,176,32,0.16)"},
                {"range": [8, 30], "color": "rgba(0,214,143,0.16)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.8, "value": 8},
        },
    ))
    return _base_layout(fig, 240)


def calibration_chart(buckets: list[dict[str, Any]]) -> go.Figure:
    if not buckets:
        return _base_layout(go.Figure(), 260)
    pred = [b["predicted"] * 100 for b in buckets]
    act = [b["actual"] * 100 for b in buckets]
    labels = [b["bucket"] for b in buckets]
    fig = go.Figure()
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                    line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    fig.add_scatter(x=pred, y=act, mode="markers+lines", name="Model",
                    marker=dict(size=12, color=GREEN, line=dict(color="#04121a", width=1)),
                    text=labels, textposition="top center")
    fig.update_layout(xaxis_title="Predicted probability (%)", yaxis_title="Actual win rate (%)")
    return _base_layout(fig, 340)


def profit_curve(curve: list[dict[str, Any]]) -> go.Figure:
    if not curve:
        return _base_layout(go.Figure(), 260)
    x = [c["bet"] for c in curve]
    y = [c["cum_profit"] for c in curve]
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name="Cumulative units",
                    line=dict(color=GREEN, width=2.4), fill="tozeroy",
                    fillcolor="rgba(0,214,143,0.12)")
    fig.update_layout(xaxis_title="Bet number", yaxis_title="Cumulative profit (units)")
    return _base_layout(fig, 340)


def history_chart(rows: list[dict[str, Any]]) -> go.Figure:
    if not rows:
        return _base_layout(go.Figure(), 240)
    rows = list(reversed(rows))
    x = [str(r["created_at"])[5:16].replace("T", " ") for r in rows]
    y = [r["value_pct"] or 0 for r in rows]
    colors = [GREEN if v >= 8 else YELLOW if v >= 3 else RED for v in y]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
    fig.update_layout(xaxis_title="Analysis timestamp", yaxis_title="Value %", showlegend=False)
    return _base_layout(fig, 260)


def confidence_ring(confidence: float) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[confidence, 1 - confidence], hole=0.74, sort=False, direction="clockwise",
        marker=dict(colors=[BLUE if confidence >= 0.7 else YELLOW if confidence >= 0.5 else RED,
                            "rgba(255,255,255,0.06)"]),
        textinfo="none", hoverinfo="skip",
    ))
    fig.add_annotation(text=f"{confidence*100:.0f}<span style='font-size:12px'>%</span>",
                       showarrow=False, font=dict(size=30, color="#e8eef9"))
    fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
