"""
backtester/engine.py – Walk-forward backtester using vectorbt.
Tests the current trade setup strategy on 3–5 years of historical data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.indicators import add_all_indicators

logger = logging.getLogger(__name__)

try:
    import vectorbt as vbt  # type: ignore
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("vectorbt not installed – using manual backtester fallback.")


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    total_trades: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    buy_hold_return_pct: float = 0.0
    nifty_return_pct: float = 0.0
    is_valid: bool = False          # True if total_trades >= 30
    is_out_of_sample_valid: bool = False
    warnings: List[str] = field(default_factory=list)
    commission_pct: float = 0.03
    slippage_pct: float = 0.05


def _apply_costs(returns: np.ndarray, commission: float, slippage: float) -> np.ndarray:
    """Deduct brokerage + slippage from each trade return."""
    cost = (commission + slippage) / 100 * 2  # round-trip
    return returns - cost


def _manual_backtest(
    df: pd.DataFrame,
    direction_col: str = "signal",
    commission_pct: float = 0.03,
    slippage_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Simple manual backtest when vectorbt is unavailable.
    signal column: 1=BUY, -1=SELL, 0=hold
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_dir = 0

    for i in range(1, len(df)):
        sig = df[direction_col].iloc[i - 1]
        price = float(df["Close"].iloc[i])

        if not in_trade and sig != 0:
            in_trade = True
            entry_price = price * (1 + slippage_pct / 100)
            entry_dir = int(sig)

        elif in_trade:
            # Exit after N bars or on reversal signal
            exit_price = price * (1 - slippage_pct / 100)
            raw_ret = (exit_price - entry_price) / entry_price * entry_dir
            net_ret = raw_ret - (commission_pct + slippage_pct) / 100 * 2
            trades.append(net_ret)
            in_trade = False

    if not trades:
        return {}

    arr = np.array(trades)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    return {
        "total_trades": len(arr),
        "win_rate": len(wins) / len(arr) if arr.size else 0,
        "avg_win": float(wins.mean()) if wins.size else 0,
        "avg_loss": float(losses.mean()) if losses.size else 0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else 0,
        "expectancy": float(arr.mean()),
        "total_return": float(arr.sum()),
    }


def generate_ema_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Generate BUY(1)/SELL(-1)/hold(0) signal series based on EMA crossover."""
    df = add_all_indicators(df, cfg)
    signal = pd.Series(0, index=df.index)
    ema20 = df.get("EMA_20")
    ema50 = df.get("EMA_50")
    if ema20 is None or ema50 is None:
        return signal
    # Use crossover: signal fires only on the bar the cross occurs
    bullish_cross = (ema20 > ema50) & (ema20.shift(1) <= ema50.shift(1))
    bearish_cross = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))
    signal[bullish_cross] = 1
    signal[bearish_cross] = -1
    return signal


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    strategy_name: str = "EMA_Crossover",
    commission_pct: float = 0.03,
    slippage_pct: float = 0.05,
    cfg: Dict[str, Any] | None = None,
) -> BacktestResult:
    """
    Run a walk-forward backtest.
    Splits: 70% in-sample, 30% out-of-sample.
    """
    result = BacktestResult(
        symbol=symbol,
        strategy_name=strategy_name,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )

    if df.empty or len(df) < 60:
        result.warnings.append("Insufficient data for backtest (need ≥60 bars).")
        return result

    cfg = cfg or {}

    # Compute buy-hold
    first_price = float(df["Close"].iloc[0])
    last_price = float(df["Close"].iloc[-1])
    years = len(df) / 252
    bh_return = (last_price - first_price) / first_price * 100
    bh_cagr = ((last_price / first_price) ** (1 / years) - 1) * 100 if years > 0 else 0
    result.buy_hold_return_pct = round(bh_return, 2)

    # Walk-forward split
    split_idx = int(len(df) * 0.7)
    df_in = df.iloc[:split_idx].copy()
    df_out = df.iloc[split_idx:].copy()

    def _run_one(df_part: pd.DataFrame) -> Dict[str, Any]:
        signals = generate_ema_signals(df_part, cfg)
        df_part = df_part.copy()
        df_part["signal"] = signals
        return _manual_backtest(df_part, commission_pct=commission_pct, slippage_pct=slippage_pct)

    stats_in = _run_one(df_in)
    stats_out = _run_one(df_out)

    if not stats_in:
        result.warnings.append("Backtest returned no trades (in-sample).")
        return result

    result.total_trades = stats_in.get("total_trades", 0)
    result.win_rate = round(stats_in.get("win_rate", 0) * 100, 1)
    result.profit_factor = round(stats_in.get("profit_factor", 0), 2)
    result.expectancy = round(stats_in.get("expectancy", 0) * 100, 2)  # in %
    result.total_return_pct = round(stats_in.get("total_return", 0) * 100, 2)

    result.cagr = round(bh_cagr, 2)  # buy-hold CAGR proxy stored separately
    result.buy_hold_return_pct = round(bh_return, 2)

    # Compute actual strategy CAGR from in-sample cumulative returns
    signals_in = generate_ema_signals(df_in, cfg)
    strat_daily = signals_in.shift(1).fillna(0) * df_in["Close"].pct_change()
    strat_total_ret = float((1 + strat_daily).prod() - 1)
    in_years = len(df_in) / 252
    strat_cagr = ((1 + strat_total_ret) ** (1 / in_years) - 1) * 100 if in_years > 0 else 0.0
    result.cagr = round(strat_cagr, 2)

    # Max drawdown from strategy equity curve
    pnl_curve = (1 + strat_daily).cumprod()
    roll_max = pnl_curve.cummax()
    dd = (pnl_curve - roll_max) / roll_max
    result.max_drawdown = round(float(dd.min()) * 100, 2)

    # Sharpe (annualised, risk-free = 6.5%)
    rf_daily = 0.065 / 252
    excess = strat_daily - rf_daily
    result.sharpe = round(float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252)), 2)

    # Validity checks
    result.is_valid = result.total_trades >= 30
    result.is_out_of_sample_valid = (
        stats_out.get("total_trades", 0) >= 5
        and stats_out.get("expectancy", 0) > 0
    )

    if not result.is_valid:
        result.warnings.append(f"Only {result.total_trades} trades – need ≥30 for valid backtest.")
    if not result.is_out_of_sample_valid:
        result.warnings.append("Out-of-sample expectancy is negative or insufficient.")

    return result
