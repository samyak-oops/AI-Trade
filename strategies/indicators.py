"""
strategies/indicators.py – Compute all technical indicators using the `ta` library.
Falls back to manual pandas calculations where needed.
All functions return a copy of the DataFrame with new indicator columns added.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try ta library first, fall back to manual computations
try:
    import ta  # type: ignore
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not installed – using pure-pandas indicators.")


# ── Pure-pandas helpers ────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    width = (upper - lower) / mid
    pct = (series - lower) / (upper - lower + 1e-9)
    return lower, mid, upper, width, pct


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    atr = _atr(high, low, close, period)
    plus_di = 100 * _ema(plus_dm, period) / (atr + 1e-9)
    minus_di = 100 * _ema(minus_dm, period) / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = _ema(dx, period)
    return adx, plus_di, minus_di


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 7, multiplier: float = 3.0):
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        # Adjust bands
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or close.iloc[i - 1] < supertrend.iloc[i - 1]:
            lb = lower_band.iloc[i]
        else:
            lb = max(lower_band.iloc[i], supertrend.iloc[i - 1]) if not np.isnan(supertrend.iloc[i - 1]) else lower_band.iloc[i]

        if upper_band.iloc[i] < upper_band.iloc[i - 1] or close.iloc[i - 1] > supertrend.iloc[i - 1]:
            ub = upper_band.iloc[i]
        else:
            ub = min(upper_band.iloc[i], supertrend.iloc[i - 1]) if not np.isnan(supertrend.iloc[i - 1]) else upper_band.iloc[i]

        if np.isnan(supertrend.iloc[i - 1]):
            supertrend.iloc[i] = lb
            direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            if close.iloc[i] > ub:
                supertrend.iloc[i] = lb
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = ub
                direction.iloc[i] = -1
        else:
            if close.iloc[i] < lb:
                supertrend.iloc[i] = ub
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lb
                direction.iloc[i] = 1

    return supertrend, direction


def add_all_indicators(df: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Master function: adds all indicators to OHLCV DataFrame.
    cfg: dict from config.yaml['strategies']
    """
    if df.empty or len(df) < 30:
        logger.warning("DataFrame too short for indicator calculation (%d rows)", len(df))
        return df

    c = cfg or {}
    df = df.copy()

    close = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    high = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
    low = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
    volume = df["Volume"].iloc[:, 0] if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]

    # ── EMA ───────────────────────────────────────────────────────────────────
    for p in c.get("ema_periods", [20, 50, 200]):
        df[f"EMA_{p}"] = _ema(close, p)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI"] = _rsi(close, c.get("rsi_period", 14))

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = _macd(
        close,
        fast=c.get("macd_fast", 12),
        slow=c.get("macd_slow", 26),
        signal=c.get("macd_signal", 9),
    )
    df["MACD"] = macd_line
    df["MACD_signal"] = macd_sig
    df["MACD_hist"] = macd_hist

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_lower, bb_mid, bb_upper, bb_width, bb_pct = _bollinger(
        close,
        period=c.get("bb_period", 20),
        std=c.get("bb_std", 2),
    )
    df["BB_lower"] = bb_lower
    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_upper
    df["BB_width"] = bb_width
    df["BB_pct"] = bb_pct

    # ── ATR ───────────────────────────────────────────────────────────────────
    df["ATR"] = _atr(high, low, close, period=14)

    # ── ADX ───────────────────────────────────────────────────────────────────
    adx, plus_di, minus_di = _adx(high, low, close, c.get("adx_period", 14))
    df["ADX"] = adx
    df["DMP"] = plus_di
    df["DMN"] = minus_di

    # ── Supertrend ────────────────────────────────────────────────────────────
    st, st_dir = _supertrend(
        high, low, close,
        period=c.get("supertrend_period", 7),
        multiplier=c.get("supertrend_multiplier", 3),
    )
    df["Supertrend"] = st
    df["Supertrend_dir"] = st_dir

    # ── Volume MA ─────────────────────────────────────────────────────────────
    df["Volume_MA20"] = volume.rolling(20).mean()
    df["Volume_ratio"] = volume / (df["Volume_MA20"] + 1e-9)

    # ── Donchian Channel (52-wk high/low on daily data → 260 bars) ───────────
    donchian_bars = c.get("donchian_period", 52) * 5
    df["Donchian_high"] = high.rolling(donchian_bars).max()
    df["Donchian_low"] = low.rolling(donchian_bars).min()

    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    """
    Detect common candlestick patterns on the last 3 candles.
    Returns list of triggered pattern names.
    """
    if df.empty or len(df) < 3:
        return []

    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    o, h, l, c = last["Open"], last["High"], last["Low"], last["Close"]
    po, ph, pl, pc = prev["Open"], prev["High"], prev["Low"], prev["Close"]

    body = abs(c - o)
    wick_upper = h - max(o, c)
    wick_lower = min(o, c) - l

    # Hammer
    if wick_lower > 2 * body and wick_upper < body * 0.3:
        patterns.append("Hammer")
    # Inverted Hammer
    if wick_upper > 2 * body and wick_lower < body * 0.3:
        patterns.append("Inverted Hammer")
    # Doji
    if body < (h - l) * 0.1:
        patterns.append("Doji")
    # Bullish Engulfing
    if pc < po and c > o and c > po and o < pc:
        patterns.append("Bullish Engulfing")
    # Bearish Engulfing
    if pc > po and c < o and c < po and o > pc:
        patterns.append("Bearish Engulfing")
    # Shooting Star
    if wick_upper > 2 * body and wick_lower < body * 0.3 and c < o:
        patterns.append("Shooting Star")

    return patterns


def compute_pivots(df: pd.DataFrame) -> Dict[str, float]:
    """Compute classic floor pivot points from previous day OHLC."""
    if df.empty or len(df) < 2:
        return {}
    prev = df.iloc[-2]
    H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    P = (H + L + C) / 3
    return {
        "pivot": round(P, 2),
        "R1": round(2 * P - L, 2),
        "R2": round(P + (H - L), 2),
        "R3": round(H + 2 * (P - L), 2),
        "S1": round(2 * P - H, 2),
        "S2": round(P - (H - L), 2),
        "S3": round(L - 2 * (H - P), 2),
    }


def compute_fibonacci(swing_high: float, swing_low: float) -> Dict[str, float]:
    """Compute Fibonacci retracement levels."""
    diff = swing_high - swing_low
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    return {f"Fib_{int(l*1000)}": round(swing_high - diff * l, 2) for l in levels}
