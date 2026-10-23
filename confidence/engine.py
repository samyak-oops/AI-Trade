"""
confidence/engine.py – Deterministic confidence score engine.
The LLM NEVER assigns the score. This module computes it from real data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

from backtester.engine import BacktestResult
from strategies.setup_engine import TradeSetup

logger = logging.getLogger(__name__)

THRESHOLD = 75.0  # HARD GATE (can be configured in config.yaml)


def _load_cfg() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("confidence", {})


def _load_weights() -> Dict[str, float]:
    conf_cfg = _load_cfg()
    global THRESHOLD
    THRESHOLD = float(conf_cfg.get("threshold", 75.0))
    return conf_cfg.get("weights", {
        "indicator_confluence": 0.25,
        "multi_timeframe":       0.15,
        "backtest_edge":         0.25,
        "market_regime":         0.10,
        "liquidity_quality":     0.05,
        "fundamentals_news":     0.10,
        "agent_track_record":    0.10,
    })


@dataclass
class ConfidenceReport:
    """Full confidence computation result."""
    score: float = 0.0
    is_above_threshold: bool = False
    breakdown: Dict[str, float] = field(default_factory=dict)
    raw_subscores: Dict[str, float] = field(default_factory=dict)
    reasons_low: List[str] = field(default_factory=list)
    reasons_high: List[str] = field(default_factory=list)
    hard_rule_failures: List[str] = field(default_factory=list)
    is_calibrated: bool = False
    calibrated_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def passed_gate(self) -> bool:
        """True only if score > THRESHOLD AND no hard rule failures."""
        return self.is_above_threshold and len(self.hard_rule_failures) == 0


def compute_indicator_confluence(setup: TradeSetup) -> float:
    """
    Score 0–1 based on how many indicators agree on direction.
    """
    factors = setup.score_factors
    if not factors:
        return 0.0
    active = [v for v in factors.values() if v > 0]
    total = len(factors)
    return len(active) / total if total else 0.0


def compute_multi_timeframe(setup: TradeSetup) -> float:
    """
    Score 0–1. Penalises signals against higher-timeframe trend.
    Returns 1.0 (neutral/unconstrained) when weekly data was unavailable.
    """
    if any("Weekly trend" in w for w in setup.warnings):
        return 0.2   # heavy penalty for HTF conflict
    return 1.0       # aligned or no weekly data – full score


def compute_backtest_edge(bt: BacktestResult) -> float:
    """
    Score 0–1 based on backtest quality.
    """
    if not bt.is_valid:
        return 0.0
    score = 0.0
    if bt.win_rate >= 55:
        score += 0.3
    elif bt.win_rate >= 45:
        score += 0.15
    if bt.profit_factor >= 2.0:
        score += 0.3
    elif bt.profit_factor >= 1.5:
        score += 0.15
    if bt.expectancy > 0:
        score += 0.2
    if bt.is_out_of_sample_valid:
        score += 0.2
    return min(score, 1.0)


def compute_market_regime(
    nifty_data: Dict[str, Any],
    vix_data: Dict[str, Any],
    fii_data: Dict[str, Any],
) -> float:
    """
    Score 0–1 based on Nifty trend, VIX, FII/DII flows.
    """
    score = 0.5  # default neutral
    if nifty_data.get("change_pct", 0) > 0:
        score += 0.15
    vix = vix_data.get("vix")
    if vix is not None:
        if vix < 15:
            score += 0.15
        elif vix > 25:
            score -= 0.2
    if fii_data.get("data") is None:
        score -= 0.1  # penalise missing data
    return max(0.0, min(score, 1.0))


def compute_liquidity_quality(
    price_data: Dict[str, Any],
    fundamentals: Dict[str, Any],
) -> float:
    """
    Score 0–1. Penalises low market cap, low volume, penny stocks.
    """
    market_cap = fundamentals.get("market_cap", 0) or 0
    volume = price_data.get("volume", 0) or 0
    price = price_data.get("price", 0) or 0

    score = 1.0
    if price < 10:
        score -= 0.5   # penny stock
    if market_cap < 1_000_000_000:   # < 100 Cr (approx)
        score -= 0.3
    elif market_cap < 10_000_000_000:  # < 1000 Cr
        score -= 0.1
    if volume < 50_000:
        score -= 0.2
    return max(0.0, min(score, 1.0))


def compute_fundamentals_news(
    fundamentals: Dict[str, Any],
    news_sentiment: Dict[str, Any],
) -> float:
    """
    Score 0–1. Penalises bad fundamentals or negative news.
    """
    score = 0.7  # base
    pe = fundamentals.get("pe_ratio")
    if pe is not None and pe > 0:
        if 10 <= pe <= 40:
            score += 0.1
        elif pe > 80:
            score -= 0.2
    sentiment = news_sentiment.get("label", "neutral")
    if sentiment == "bullish":
        score += 0.2
    elif sentiment == "bearish":
        score -= 0.2
    if fundamentals.get("error"):
        score -= 0.1
    return max(0.0, min(score, 1.0))


def compute_agent_track_record(past_signal_stats: Dict[str, Any]) -> float:
    """
    Score 0–1 based on agent's own historical signal accuracy (from DB).
    Falls back to 0.5 (neutral) if there are no past signals.
    """
    n = past_signal_stats.get("total", 0)
    if n < 5:
        return 0.5   # uncalibrated
    hit_rate = past_signal_stats.get("hit_rate", 0.5)
    return min(hit_rate, 1.0)


def compute_confidence(
    setup: TradeSetup,
    bt: BacktestResult,
    market_data: Dict[str, Any],
    fundamentals: Dict[str, Any],
    news_sentiment: Dict[str, Any],
    past_signal_stats: Dict[str, Any],
    price_data: Dict[str, Any],
    fii_data: Dict[str, Any] | None = None,
) -> ConfidenceReport:
    """
    Master confidence computation. Fully deterministic – no LLM involvement.
    Returns ConfidenceReport with score, breakdown, hard rule failures.
    """
    weights = _load_weights()
    report = ConfidenceReport()

    if setup.direction == "NO_SIGNAL":
        report.reasons_low.append("No clear directional signal from indicators.")
        return report

    # ── Compute sub-scores ────────────────────────────────────────────────────
    s_confluence = compute_indicator_confluence(setup)
    s_mtf = compute_multi_timeframe(setup)
    s_backtest = compute_backtest_edge(bt)
    s_regime = compute_market_regime(
        market_data.get("nifty", {}),
        market_data.get("vix", {}),
        fii_data or {},   # pass fii_data directly (it's a top-level state key)
    )
    s_liquidity = compute_liquidity_quality(price_data, fundamentals)
    s_fundam = compute_fundamentals_news(fundamentals, news_sentiment)
    s_track = compute_agent_track_record(past_signal_stats)

    report.raw_subscores = {
        "indicator_confluence": round(s_confluence, 3),
        "multi_timeframe":       round(s_mtf, 3),
        "backtest_edge":         round(s_backtest, 3),
        "market_regime":         round(s_regime, 3),
        "liquidity_quality":     round(s_liquidity, 3),
        "fundamentals_news":     round(s_fundam, 3),
        "agent_track_record":    round(s_track, 3),
    }

    # Weighted score (0–100)
    raw = (
        s_confluence * weights.get("indicator_confluence", 0.25)
        + s_mtf       * weights.get("multi_timeframe",       0.15)
        + s_backtest  * weights.get("backtest_edge",         0.25)
        + s_regime    * weights.get("market_regime",         0.10)
        + s_liquidity * weights.get("liquidity_quality",     0.05)
        + s_fundam    * weights.get("fundamentals_news",     0.10)
        + s_track     * weights.get("agent_track_record",    0.10)
    )
    report.score = round(raw * 100, 1)
    report.breakdown = {
        k: round(v * weights.get(k, 0) * 100, 1)
        for k, v in report.raw_subscores.items()
    }

    # ── Hard rules ────────────────────────────────────────────────────────────
    if setup.risk_reward < 2.0:
        report.hard_rule_failures.append(
            f"Risk:Reward is {setup.risk_reward:.2f} (required ≥2.0)"
        )
    if not bt.is_valid:
        report.hard_rule_failures.append(
            f"Backtest has only {bt.total_trades} trades (required ≥30)"
        )
    if not bt.is_out_of_sample_valid:
        report.hard_rule_failures.append("Out-of-sample expectancy is negative.")

    if any("Weekly trend" in w for w in setup.warnings):
        report.hard_rule_failures.append(
            "Signal conflicts with higher-timeframe (weekly) trend."
        )

    report.is_above_threshold = report.score > THRESHOLD

    # ── Reasons ───────────────────────────────────────────────────────────────
    for key, sub in report.raw_subscores.items():
        label = key.replace("_", " ").title()
        if sub < 0.4:
            report.reasons_low.append(f"{label}: {sub:.0%}")
        elif sub > 0.7:
            report.reasons_high.append(f"{label}: {sub:.0%}")

    # ── Calibration placeholder ───────────────────────────────────────────────
    n_signals = past_signal_stats.get("total", 0)
    if n_signals < 20:
        report.is_calibrated = False
    else:
        report.is_calibrated = True
        report.calibrated_score = report.score  # overwrite with isotonic result

    return report
