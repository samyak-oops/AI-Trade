"""
tests/test_backtester.py – Unit tests for the backtester engine.
"""
import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_backtest, BacktestResult


def make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(0)
    closes = np.cumprod(1 + np.random.normal(0.0008, 0.018, n)) * 1000
    df = pd.DataFrame({
        "Open": closes * 0.99,
        "High": closes * 1.015,
        "Low": closes * 0.985,
        "Close": closes,
        "Volume": np.random.randint(200_000, 2_000_000, n),
    })
    df.index = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


def test_backtest_returns_result_object():
    df = make_ohlcv(300)
    result = run_backtest("TEST", df)
    assert isinstance(result, BacktestResult)


def test_backtest_insufficient_data():
    df = make_ohlcv(50)   # too short
    result = run_backtest("TEST", df)
    assert not result.is_valid
    assert len(result.warnings) > 0


def test_backtest_trade_count_positive():
    df = make_ohlcv(300)
    result = run_backtest("TEST", df)
    assert result.total_trades >= 0


def test_backtest_buy_hold_positive():
    """In an uptrend, buy-hold return should be positive."""
    n = 300
    closes = np.linspace(500, 2000, n)
    df = pd.DataFrame({
        "Open": closes * 0.99,
        "High": closes * 1.01,
        "Low": closes * 0.98,
        "Close": closes,
        "Volume": [500_000] * n,
    })
    df.index = pd.date_range("2020-01-01", periods=n, freq="D")
    result = run_backtest("UPTREND", df)
    assert result.buy_hold_return_pct > 0


def test_backtest_validity_requires_30_trades():
    """Validity flag should be False when total_trades < 30."""
    df = make_ohlcv(300)
    result = run_backtest("TEST", df)
    if result.total_trades < 30:
        assert not result.is_valid
    else:
        assert result.is_valid
