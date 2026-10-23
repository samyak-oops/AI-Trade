"""
strategies/setup_engine.py – Trade setup engine.
Evaluates all strategy signals and produces entry/exit/SL/target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from strategies.indicators import add_all_indicators, detect_candlestick_patterns, compute_pivots

logger = logging.getLogger(__name__)


def _load_cfg() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("strategies", {})


@dataclass
class TradeSetup:
    """Proposed trade setup produced by the engine."""
    symbol: str
    direction: str                          # "BUY" | "SELL" | "NO_SIGNAL"
    timeframe: str                          # "intraday" | "swing" | "positional"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    risk_reward: float = 0.0
    atr: float = 0.0
    strategies_triggered: List[str] = field(default_factory=list)
    candlestick_patterns: List[str] = field(default_factory=list)
    pivots: Dict[str, float] = field(default_factory=dict)
    score_factors: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    regime: str = "UNKNOWN"                 # "TREND" | "RANGE" | "TRANSITIONAL"
    families_triggered: List[str] = field(default_factory=list)
    eligible_instrument: str = "DELIVERY"   # "DELIVERY" | "INTRADAY_OR_FNO"

    def is_valid(self) -> bool:
        return (
            self.direction in ("BUY", "SELL")
            and self.entry_price > 0
            and self.stop_loss > 0
            and self.risk_reward >= 2.0
        )


def analyze_setup(
    symbol: str,
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame | None = None,
    df_intraday: pd.DataFrame | None = None,
    current_price: float | None = None,
) -> TradeSetup:
    """
    Main entry: given OHLCV DataFrames (daily + optional weekly/intraday),
    compute indicators, detect strategies, produce a TradeSetup.
    """
    cfg = _load_cfg()

    if df_daily.empty or len(df_daily) < 50:
        return TradeSetup(symbol=symbol, direction="NO_SIGNAL",
                          timeframe="swing", warnings=["Insufficient historical data"])

    # ── Add indicators ────────────────────────────────────────────────────────
    df = add_all_indicators(df_daily, cfg)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    price = current_price or float(last["Close"])

    bullish_triggers: List[str] = []
    bearish_triggers: List[str] = []
    neutral_triggers: List[str] = []
    direction_votes = {"BUY": 0, "SELL": 0}
    score_factors: Dict[str, float] = {}

    # ── 1. EMA Trend ──────────────────────────────────────────────────────────
    ema20 = last.get("EMA_20")
    ema50 = last.get("EMA_50")
    ema200 = last.get("EMA_200")
    ema_score = 0.0
    if all(v is not None and not np.isnan(v) for v in [ema20, ema50, ema200]):
        if price > ema20 > ema50 > ema200:
            direction_votes["BUY"] += 2
            ema_score = 1.0
            bullish_triggers.append("EMA Golden Alignment (Price > 20 > 50 > 200)")
        elif price < ema20 < ema50 < ema200:
            direction_votes["SELL"] += 2
            ema_score = 1.0
            bearish_triggers.append("EMA Bearish Alignment (Price < 20 < 50 < 200)")
        elif price > ema50:
            direction_votes["BUY"] += 1
            ema_score = 0.5
            bullish_triggers.append("Price Above 50 EMA")
        elif price < ema50:
            direction_votes["SELL"] += 1
            ema_score = 0.5
            bearish_triggers.append("Price Below 50 EMA")
    score_factors["ema_trend"] = ema_score

    # ── 2. RSI Momentum ───────────────────────────────────────────────────────
    rsi = last.get("RSI")
    prev_rsi = prev.get("RSI")
    rsi_score = 0.0
    if rsi is not None and not np.isnan(rsi):
        if rsi < cfg.get("rsi_oversold", 30) and prev_rsi is not None and rsi > prev_rsi:
            direction_votes["BUY"] += 2
            rsi_score = 1.0
            bullish_triggers.append(f"RSI Oversold Reversal ({rsi:.1f})")
        elif rsi > cfg.get("rsi_overbought", 70) and prev_rsi is not None and rsi < prev_rsi:
            direction_votes["SELL"] += 2
            rsi_score = 1.0
            bearish_triggers.append(f"RSI Overbought Reversal ({rsi:.1f})")
        elif 40 <= rsi <= 60:
            rsi_score = 0.5  # neutral
        else:
            rsi_score = 0.3
    score_factors["rsi"] = rsi_score

    # ── 3. MACD Crossover ─────────────────────────────────────────────────────
    macd = last.get("MACD")
    macd_sig = last.get("MACD_signal")
    macd_hist = last.get("MACD_hist")
    prev_hist = prev.get("MACD_hist")
    macd_score = 0.0
    if all(v is not None and not np.isnan(v) for v in [macd, macd_sig, macd_hist]):
        if (prev_hist is not None and not np.isnan(prev_hist)
                and prev_hist < 0 <= macd_hist):
            direction_votes["BUY"] += 2
            macd_score = 1.0
            bullish_triggers.append("MACD Bullish Crossover")
        elif (prev_hist is not None and not np.isnan(prev_hist)
              and prev_hist > 0 >= macd_hist):
            direction_votes["SELL"] += 2
            macd_score = 1.0
            bearish_triggers.append("MACD Bearish Crossover")
        elif macd > macd_sig and macd > 0:
            direction_votes["BUY"] += 1
            macd_score = 0.5
            bullish_triggers.append("MACD Positive Momentum")
        elif macd < macd_sig and macd < 0:
            direction_votes["SELL"] += 1
            macd_score = 0.5
            bearish_triggers.append("MACD Negative Momentum")
    score_factors["macd"] = macd_score

    # ── 4. ADX (trend strength filter) ───────────────────────────────────────
    adx = last.get("ADX")
    adx_threshold = cfg.get("adx_trend_threshold", 25)
    adx_score = 0.0
    if adx is not None and not np.isnan(adx):
        adx_score = min(float(adx) / 50, 1.0)
        if adx > adx_threshold:
            neutral_triggers.append(f"Strong Trend (ADX={adx:.1f})")
            score_factors["adx"] = adx_score
        else:
            score_factors["adx"] = adx_score * 0.5

    # ── 5. Supertrend ─────────────────────────────────────────────────────────
    st_dir = last.get("Supertrend_dir")
    st_score = 0.0
    if st_dir is not None and not np.isnan(st_dir):
        if st_dir == 1:
            direction_votes["BUY"] += 2
            st_score = 1.0
            bullish_triggers.append("Supertrend Bullish")
        elif st_dir == -1:
            direction_votes["SELL"] += 2
            st_score = 1.0
            bearish_triggers.append("Supertrend Bearish")
    score_factors["supertrend"] = st_score

    # ── 6. Breakout (Donchian / 52-wk high & low) ───────────────────────────
    d_high = last.get("Donchian_high")
    d_low = last.get("Donchian_low")
    vol_ratio = last.get("Volume_ratio")
    vol_surge = cfg.get("volume_surge_multiplier", 1.5)
    breakout_score = 0.0
    if d_high and not np.isnan(d_high):
        if price >= d_high and vol_ratio and vol_ratio >= vol_surge:
            direction_votes["BUY"] += 3
            breakout_score = 1.0
            bullish_triggers.append(f"52-Wk High Breakout (Vol ratio={vol_ratio:.1f}x)")
    if d_low and not np.isnan(d_low):
        if price <= d_low and vol_ratio and vol_ratio >= vol_surge:
            direction_votes["SELL"] += 3
            breakout_score = 1.0
            bearish_triggers.append(f"52-Wk Low Breakdown (Vol ratio={vol_ratio:.1f}x)")
    score_factors["breakout"] = breakout_score

    # ── 7. Bollinger Bands mean reversion ─────────────────────────────────────
    bb_lower = last.get("BB_lower")
    bb_upper = last.get("BB_upper")
    bb_score = 0.0
    if all(v is not None and not np.isnan(v) for v in [bb_lower, bb_upper, rsi]):
        if price <= bb_lower and rsi < 35:
            direction_votes["BUY"] += 2
            bb_score = 1.0
            bullish_triggers.append("BB + RSI Oversold Reversal")
        elif price >= bb_upper and rsi > 65:
            direction_votes["SELL"] += 2
            bb_score = 1.0
            bearish_triggers.append("BB + RSI Overbought Reversal")
    score_factors["bollinger"] = bb_score

    # ── Regime Determination (ADX 14) ─────────────────────────────────────────
    adx_val = float(adx) if adx is not None and not np.isnan(adx) else 20.0
    if adx_val > 25:
        detected_regime = "TREND"
    elif adx_val < 20:
        detected_regime = "RANGE"
    else:
        detected_regime = "TRANSITIONAL"

    # ── Signal Families (count each family once) ──────────────────────────────
    # Families: 'Trend', 'Momentum/Breakout', 'Mean-Reversion'
    bullish_families: List[str] = []
    bearish_families: List[str] = []

    # Trend family: EMA, Supertrend, MACD
    if any(k in " ".join(bullish_triggers) for k in ["EMA", "Supertrend", "MACD"]):
        bullish_families.append("Trend Family (EMA/Supertrend/MACD)")
    if any(k in " ".join(bearish_triggers) for k in ["EMA", "Supertrend", "MACD"]):
        bearish_families.append("Trend Family (EMA/Supertrend/MACD)")

    # Momentum/Breakout family: 52-wk high/low breakout with volume
    if any("52-Wk High Breakout" in t for t in bullish_triggers):
        bullish_families.append("Momentum/Breakout Family (52-Wk High)")
    if any("52-Wk Low Breakdown" in t for t in bearish_triggers):
        bearish_families.append("Momentum/Breakout Family (52-Wk Low)")

    # Mean-Reversion family: RSI, Bollinger
    if any(k in " ".join(bullish_triggers) for k in ["Oversold", "BB + RSI"]):
        bullish_families.append("Mean-Reversion Family (RSI/Bollinger)")
    if any(k in " ".join(bearish_triggers) for k in ["Overbought", "BB + RSI"]):
        bearish_families.append("Mean-Reversion Family (RSI/Bollinger)")

    # Filter families by regime
    # Trend regime: only Trend & Breakout setups eligible
    # Range regime: only Mean-Reversion setups eligible
    eligible_bullish_families = [
        f for f in bullish_families
        if (detected_regime == "TREND" and "Mean-Reversion" not in f)
        or (detected_regime == "RANGE" and "Mean-Reversion" in f)
        or (detected_regime == "TRANSITIONAL")
    ]
    eligible_bearish_families = [
        f for f in bearish_families
        if (detected_regime == "TREND" and "Mean-Reversion" not in f)
        or (detected_regime == "RANGE" and "Mean-Reversion" in f)
        or (detected_regime == "TRANSITIONAL")
    ]

    # Require agreement from at least 2 independent families in allowed regime
    regime_warnings: List[str] = []
    if detected_regime == "TRANSITIONAL":
        regime_warnings.append(f"Regime is Transitional (ADX={adx_val:.1f} is between 20-25) – higher uncertainty.")

    if len(eligible_bullish_families) >= 2 and len(eligible_bullish_families) > len(eligible_bearish_families):
        direction = "BUY"
        strategies_triggered = bullish_triggers + neutral_triggers
        active_families = eligible_bullish_families
        instrument = "DELIVERY"
    elif len(eligible_bearish_families) >= 2 and len(eligible_bearish_families) > len(eligible_bullish_families):
        direction = "SELL"
        strategies_triggered = bearish_triggers + neutral_triggers
        active_families = eligible_bearish_families
        instrument = "INTRADAY_OR_FNO"
        regime_warnings.append("SELL/Short setup: Equity cash delivery short is prohibited by SEBI; execute as Intraday MIS or F&O only.")
    else:
        direction = "NO_SIGNAL"
        strategies_triggered = []
        active_families = []
        instrument = "DELIVERY"

    # ── ATR-based stop-loss & targets ───────────────────────────────────────
    atr = float(last.get("ATR", 0) or 0)
    atr_sl_mult = cfg.get("atr_multiplier_sl", 2.0)   # SL distance = 2×ATR
    atr_t1_mult = atr_sl_mult * 2                      # T1 = 4×ATR → 2:1 R:R
    atr_t2_mult = atr_sl_mult * 3                      # T2 = 6×ATR → 3:1 R:R
    if direction == "BUY":
        sl = price - atr_sl_mult * atr
        t1 = price + atr_t1_mult * atr
        t2 = price + atr_t2_mult * atr
    elif direction == "SELL":
        sl = price + atr_sl_mult * atr
        t1 = price - atr_t1_mult * atr
        t2 = price - atr_t2_mult * atr
    else:
        sl = t1 = t2 = price

    risk = abs(price - sl)
    reward = abs(t1 - price)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    # ── Candlestick patterns ──────────────────────────────────────────────────
    patterns = []
    try:
        patterns = detect_candlestick_patterns(df)
    except Exception as exc:
        logger.warning("Candlestick detection failed: %s", exc)

    # ── Pivots ────────────────────────────────────────────────────────────────
    pivots = compute_pivots(df)

    # ── Multi-timeframe check ─────────────────────────────────────────────────
    warnings = []
    if direction == "BUY" and bearish_triggers:
        for bt in bearish_triggers:
            warnings.append(f"Caution: Counter-trend indicator: {bt}")
    elif direction == "SELL" and bullish_triggers:
        for bt in bullish_triggers:
            warnings.append(f"Caution: Counter-trend indicator: {bt}")

    if df_weekly is not None and not df_weekly.empty:
        dfw = add_all_indicators(df_weekly, cfg)
        weekly_ema50 = dfw.iloc[-1].get("EMA_50")
        weekly_last_close = float(dfw.iloc[-1]["Close"])
        if direction == "BUY" and weekly_ema50 and weekly_last_close < weekly_ema50:
            warnings.append("Weekly trend is bearish – BUY signal against higher-timeframe trend.")
        elif direction == "SELL" and weekly_ema50 and weekly_last_close > weekly_ema50:
            warnings.append("Weekly trend is bullish – SELL signal against higher-timeframe trend.")

    # Timeframe heuristic
    timeframe = "swing"
    if df_intraday is not None:
        timeframe = "intraday"

    return TradeSetup(
        symbol=symbol,
        direction=direction,
        timeframe=timeframe,
        entry_price=round(price, 2),
        stop_loss=round(sl, 2),
        target1=round(t1, 2),
        target2=round(t2, 2),
        risk_reward=rr,
        atr=round(atr, 2),
        strategies_triggered=strategies_triggered,
        candlestick_patterns=patterns,
        pivots=pivots,
        score_factors=score_factors,
        warnings=regime_warnings + warnings,
        regime=detected_regime,
        families_triggered=active_families,
        eligible_instrument=instrument,
    )
