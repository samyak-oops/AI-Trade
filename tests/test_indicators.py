"""
tests/test_indicators.py – Unit tests for indicator computation.
"""
import numpy as np
import pandas as pd
import pytest

from strategies.indicators import add_all_indicators, compute_pivots, compute_fibonacci


def make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    closes = np.cumprod(1 + np.random.normal(0.001, 0.02, n)) * 1000
    df = pd.DataFrame({
        "Open": closes * 0.99,
        "High": closes * 1.01,
        "Low": closes * 0.98,
        "Close": closes,
        "Volume": np.random.randint(100_000, 1_000_000, n),
    })
    df.index = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


def test_add_all_indicators_columns():
    df = make_ohlcv(250)
    result = add_all_indicators(df)
    assert "EMA_20" in result.columns
    assert "EMA_50" in result.columns
    assert "EMA_200" in result.columns
    assert "RSI" in result.columns
    assert "MACD" in result.columns
    assert "ATR" in result.columns
    assert "ADX" in result.columns


def test_rsi_range():
    df = make_ohlcv(250)
    result = add_all_indicators(df)
    rsi = result["RSI"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be 0–100"


def test_ema_monotone_alignment():
    """In a strong uptrend, EMA_20 > EMA_50 at some point."""
    df = make_ohlcv(250)
    # Force uptrend
    df["Close"] = np.linspace(500, 2000, 250)
    df["Open"] = df["Close"] * 0.99
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.98
    result = add_all_indicators(df)
    last = result.iloc[-1]
    assert last["EMA_20"] > last["EMA_50"], "EMA_20 should be > EMA_50 in strong uptrend"


def test_compute_pivots():
    df = make_ohlcv(10)
    pivots = compute_pivots(df)
    assert "pivot" in pivots
    assert "R1" in pivots
    assert "S1" in pivots
    # R1 > Pivot > S1
    assert pivots["R1"] > pivots["pivot"] > pivots["S1"]


def test_fibonacci_levels():
    fibs = compute_fibonacci(swing_high=200.0, swing_low=100.0)
    assert "Fib_0" in fibs      # swing high
    assert "Fib_1000" in fibs   # swing low
    assert fibs["Fib_0"] == 200.0
    assert fibs["Fib_1000"] == 100.0
    # Fib 0.618 should be between high and low
    fib618 = fibs.get("Fib_618")
    assert fib618 is not None
    assert 100 < fib618 < 200


def test_insufficient_data_returns_df():
    df = make_ohlcv(10)   # too short
    result = add_all_indicators(df)
    assert isinstance(result, pd.DataFrame)
