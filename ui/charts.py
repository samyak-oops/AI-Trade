"""
ui/charts.py – Plotly interactive chart builders.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.setup_engine import TradeSetup


def build_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    setup: Optional[TradeSetup] = None,
    pivots: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """
    Build a full interactive chart:
      - Candlestick + volume
      - EMA 20/50/200
      - RSI subplot
      - Entry / SL / Target lines if setup is provided
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} – No data available")
        return fig

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
        subplot_titles=(symbol, "Volume", "RSI"),
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for period, color in [(20, "#f6c90e"), (50, "#fb8c00"), (200, "#ab47bc")]:
        col_name = f"EMA_{period}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col_name], name=f"EMA {period}",
                           line=dict(color=color, width=1.5)),
                row=1, col=1,
            )

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                       line=dict(color="rgba(173,216,230,0.5)", dash="dot"), showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                       line=dict(color="rgba(173,216,230,0.5)", dash="dot"),
                       fill="tonexty", fillcolor="rgba(173,216,230,0.07)"),
            row=1, col=1,
        )

    # ── Trade setup lines ─────────────────────────────────────────────────────
    if setup and setup.entry_price > 0:
        last_x = df.index[-1]
        for label, price, color in [
            ("Entry", setup.entry_price, "#29b6f6"),
            ("Stop-Loss", setup.stop_loss, "#ef5350"),
            ("Target 1", setup.target1, "#66bb6a"),
            ("Target 2", setup.target2, "#26a69a"),
        ]:
            fig.add_hline(
                y=price, line_dash="dash", line_color=color,
                annotation_text=f"{label} ₹{price:.1f}",
                annotation_position="right",
                row=1, col=1,
            )

    # ── Pivot levels ──────────────────────────────────────────────────────────
    if pivots:
        for label, price in pivots.items():
            fig.add_hline(
                y=price, line_dash="dot", line_color="rgba(200,200,200,0.4)",
                annotation_text=label, annotation_position="left",
                row=1, col=1,
            )

    # ── Volume ────────────────────────────────────────────────────────────────
    colors = ["#ef5350" if c < o else "#26a69a"
              for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    # ── RSI ───────────────────────────────────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)",
                       line=dict(color="#ce93d8", width=1.5)),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#ef5350", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#66bb6a", row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=50, r=80, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )
    return fig


def build_confidence_gauge(score: float, threshold: float = 75.0) -> go.Figure:
    """Donut / gauge chart for confidence score."""
    color = "#66bb6a" if score > threshold else "#ef5350" if score < 50 else "#ffa726"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": threshold, "increasing": {"color": "#66bb6a"},
               "decreasing": {"color": "#ef5350"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#1a1a2e"},
                {"range": [50, 75], "color": "#1e2d4e"},
                {"range": [75, 100], "color": "#1a3320"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": threshold,
            },
        },
        title={"text": "Confidence Score", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 32}},
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )
    return fig


def build_breakdown_bar(breakdown: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of confidence breakdown."""
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors = ["#66bb6a" if v >= 5 else "#ffa726" if v >= 2 else "#ef5350" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=[l.replace("_", " ").title() for l in labels],
        orientation="h", marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Confidence Breakdown (per factor)",
        template="plotly_dark",
        height=300,
        margin=dict(l=180, r=50, t=40, b=20),
        xaxis=dict(range=[0, max(values) * 1.3 if values else 30]),
        paper_bgcolor="#16213e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    return fig
