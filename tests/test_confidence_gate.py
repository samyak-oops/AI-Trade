"""
tests/test_confidence_gate.py – Critical tests for the confidence engine.
Verifies that scores ≤ 90 are always rejected, and hard rules are enforced.
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List

from confidence.engine import (
    THRESHOLD,
    ConfidenceReport,
    compute_confidence,
)
from strategies.setup_engine import TradeSetup
from backtester.engine import BacktestResult


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_perfect_setup() -> TradeSetup:
    """A trade setup that should score well."""
    return TradeSetup(
        symbol="RELIANCE",
        direction="BUY",
        timeframe="swing",
        entry_price=2500.0,
        stop_loss=2450.0,
        target1=2600.0,
        target2=2700.0,
        risk_reward=2.0,
        atr=50.0,
        strategies_triggered=["EMA Uptrend", "RSI Oversold Reversal", "MACD Bullish Crossover",
                               "Supertrend Bullish", "Strong Trend (ADX=32)"],
        score_factors={
            "ema_trend": 1.0, "rsi": 1.0, "macd": 1.0,
            "adx": 0.8, "supertrend": 1.0, "breakout": 1.0, "bollinger": 0.0,
        },
        warnings=[],
    )


def make_perfect_backtest() -> BacktestResult:
    return BacktestResult(
        symbol="RELIANCE",
        strategy_name="EMA_Crossover",
        total_trades=50,
        win_rate=60.0,
        profit_factor=2.5,
        expectancy=1.5,
        cagr=18.0,
        max_drawdown=-12.0,
        sharpe=1.8,
        total_return_pct=120.0,
        buy_hold_return_pct=80.0,
        is_valid=True,
        is_out_of_sample_valid=True,
    )


def make_market_data() -> Dict[str, Any]:
    return {
        "nifty": {"change_pct": 0.5, "price": 22000},
        "vix": {"vix": 14.0, "regime": "normal"},
        "fii": {"data": {"net_buy": 1000}, "error": None},
    }


def make_fundamentals() -> Dict[str, Any]:
    return {
        "pe_ratio": 22.0, "roe": 0.15, "market_cap": 15_000_000_000,
    }


def make_price_data(stale: bool = False) -> Dict[str, Any]:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    return {"price": 2500.0, "volume": 500_000, "timestamp": ts, "stale": stale}


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: threshold is strictly > 75.0
# ═══════════════════════════════════════════════════════════════════════════════
def test_threshold_value():
    """The threshold must be exactly 75 as per config."""
    assert THRESHOLD == 75.0


def test_score_75_is_rejected():
    """
    A score of exactly 75.0 must be rejected (threshold requires STRICTLY > 75).
    """
    report = ConfidenceReport(score=75.0)
    report.is_above_threshold = (report.score > THRESHOLD)
    assert not report.passed_gate(), "Score of 75.0 must NOT pass the gate."


def test_score_75_1_may_pass_if_no_hard_rule_failures():
    """Score of 75.1 with no hard failures should pass."""
    report = ConfidenceReport(score=75.1)
    report.is_above_threshold = True
    report.hard_rule_failures = []
    assert report.passed_gate()


def test_score_100_fails_if_hard_rule_present():
    """Even a perfect score fails if a hard rule is violated."""
    report = ConfidenceReport(score=100.0)
    report.is_above_threshold = True
    report.hard_rule_failures = ["Risk:Reward is 1.5 (required ≥2.0)"]
    assert not report.passed_gate()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: hard rule – Risk:Reward
# ═══════════════════════════════════════════════════════════════════════════════
def test_rr_below_2_triggers_hard_rule():
    setup = make_perfect_setup()
    setup.risk_reward = 1.5   # violate R:R rule
    bt = make_perfect_backtest()

    report = compute_confidence(
        setup=setup, bt=bt,
        market_data=make_market_data(),
        fundamentals=make_fundamentals(),
        news_sentiment={"label": "bullish", "score": 0.5},
        past_signal_stats={"total": 30, "hit_rate": 0.7},
        price_data=make_price_data(),
    )
    assert "Risk:Reward" in " ".join(report.hard_rule_failures)
    assert not report.passed_gate()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: hard rule – insufficient backtest trades
# ═══════════════════════════════════════════════════════════════════════════════
def test_insufficient_backtest_trades_triggers_hard_rule():
    setup = make_perfect_setup()
    bt = make_perfect_backtest()
    bt.total_trades = 20   # less than 30
    bt.is_valid = False

    report = compute_confidence(
        setup=setup, bt=bt,
        market_data=make_market_data(),
        fundamentals=make_fundamentals(),
        news_sentiment={"label": "neutral", "score": 0.0},
        past_signal_stats={"total": 30, "hit_rate": 0.7},
        price_data=make_price_data(),
    )
    assert not report.passed_gate()
    assert any("trades" in f.lower() for f in report.hard_rule_failures)


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: NO_SIGNAL direction = score 0
# ═══════════════════════════════════════════════════════════════════════════════
def test_no_signal_direction_gives_zero_score():
    setup = make_perfect_setup()
    setup.direction = "NO_SIGNAL"
    bt = make_perfect_backtest()

    report = compute_confidence(
        setup=setup, bt=bt,
        market_data=make_market_data(),
        fundamentals=make_fundamentals(),
        news_sentiment={"label": "neutral"},
        past_signal_stats={},
        price_data=make_price_data(),
    )
    assert report.score == 0.0
    assert not report.passed_gate()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: HTF conflict triggers hard rule
# ═══════════════════════════════════════════════════════════════════════════════
def test_htf_conflict_triggers_hard_rule():
    setup = make_perfect_setup()
    setup.warnings = ["⚠️ Weekly trend is bearish – BUY signal against higher-timeframe trend."]
    bt = make_perfect_backtest()

    report = compute_confidence(
        setup=setup, bt=bt,
        market_data=make_market_data(),
        fundamentals=make_fundamentals(),
        news_sentiment={"label": "neutral"},
        past_signal_stats={"total": 30, "hit_rate": 0.7},
        price_data=make_price_data(),
    )
    assert any("Weekly trend" in f or "higher-timeframe" in f
               for f in report.hard_rule_failures)
    assert not report.passed_gate()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: score breakdown sums correctly
# ═══════════════════════════════════════════════════════════════════════════════
def test_score_breakdown_sums_to_total():
    setup = make_perfect_setup()
    bt = make_perfect_backtest()

    report = compute_confidence(
        setup=setup, bt=bt,
        market_data=make_market_data(),
        fundamentals=make_fundamentals(),
        news_sentiment={"label": "bullish", "score": 0.6},
        past_signal_stats={"total": 30, "hit_rate": 0.7},
        price_data=make_price_data(),
    )
    total_from_breakdown = sum(report.breakdown.values())
    assert abs(total_from_breakdown - report.score) < 1.0, (
        f"Breakdown sum {total_from_breakdown:.1f} should ≈ score {report.score:.1f}"
    )
