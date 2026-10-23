"""
tools/equity_data.py – Live + historical equity data via yfinance + Angel One.
All functions return data with 'source' and 'timestamp' fields.
Results are cached using a simple TTL cache.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

from core.config import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

# TTL caches
_price_cache: TTLCache = TTLCache(maxsize=256, ttl=60)        # 1 min live price
_hist_cache: TTLCache = TTLCache(maxsize=64, ttl=3600)        # 1 hr historical
_fund_cache: TTLCache = TTLCache(maxsize=64, ttl=86400)       # 24 hr fundamentals


def _yf_ticker(symbol: str, exchange: str = "NSE") -> str:
    """Convert symbol to yfinance ticker (e.g. RELIANCE → RELIANCE.NS)."""
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    yaml_cfg = cfg.yaml_section("data")
    suffix_map = yaml_cfg.get("yfinance_suffix", {".NS": ".NS", ".BO": ".BO"})
    suffix = suffix_map.get(exchange, ".NS")
    return f"{symbol}{suffix}"


def get_live_price(symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
    """
    Fetch the latest price for an equity.
    Returns: price, change_pct, volume, market_cap, timestamp, source.
    Uses 1-min TTL cache.
    """
    key = f"{symbol}:{exchange}"
    if key in _price_cache:
        return _price_cache[key]

    ticker_str = _yf_ticker(symbol, exchange)
    try:
        tk = yf.Ticker(ticker_str)
        info = tk.fast_info
        price = float(info.last_price or 0)
        prev_close = float(info.previous_close or price)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        volume = int(info.three_month_average_volume or 0)
        market_cap = float(info.market_cap or 0)

        if price == 0:
            raise ValueError("Zero price returned – market may be closed.")

        result = {
            "symbol": symbol,
            "exchange": exchange,
            "price": price,
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "market_cap": market_cap,
            "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
            "stale": False,
        }
        _price_cache[key] = result
        return result

    except Exception as exc:
        logger.warning("Live price fetch failed for %s: %s", symbol, exc)
        return {
            "symbol": symbol,
            "exchange": exchange,
            "price": 0.0,
            "change_pct": 0.0,
            "volume": 0,
            "market_cap": 0,
            "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
            "stale": True,
            "error": str(exc),
        }


def get_historical_data(
    symbol: str,
    exchange: str = "NSE",
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data.
    period: '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'
    interval: '1m','5m','15m','30m','1h','1d','1wk','1mo'
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    """
    key = f"{symbol}:{exchange}:{period}:{interval}"
    if key in _hist_cache:
        return _hist_cache[key]

    ticker_str = _yf_ticker(symbol, exchange)
    try:
        df = yf.download(
            ticker_str,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"No historical data for {ticker_str}")
        df.index = pd.to_datetime(df.index)
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            if "Price" in df.columns.names:
                df.columns = df.columns.get_level_values("Price")
            else:
                for level in range(df.columns.nlevels):
                    vals = df.columns.get_level_values(level)
                    if "Close" in vals or "Open" in vals:
                        df.columns = vals
                        break
                else:
                    df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        _hist_cache[key] = df
        return df
    except Exception as exc:
        logger.error("Historical data fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def get_fundamentals(symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
    """
    Fetch fundamental data: PE, ROE, D/E, promoter holding, quarterly results.
    Uses 24h TTL cache.
    """
    key = f"fund:{symbol}:{exchange}"
    if key in _fund_cache:
        return _fund_cache[key]

    ticker_str = _yf_ticker(symbol, exchange)
    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info
        result = {
            "symbol": symbol,
            "exchange": exchange,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "book_value": info.get("bookValue"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "float_shares": info.get("floatShares"),
            "promoter_holding": None,   # yfinance doesn't expose; placeholder
            "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
        }
        # Try quarterly financials
        try:
            qf = tk.quarterly_financials
            if not qf.empty:
                if "Total Revenue" in qf.index:
                    rev_series = qf.loc["Total Revenue"].head(4)
                    result["quarterly_revenue"] = {
                        str(k)[:10]: float(v) if pd.notnull(v) else 0.0
                        for k, v in rev_series.items()
                    }
                else:
                    result["quarterly_revenue"] = {}

                if "Net Income" in qf.index:
                    inc_series = qf.loc["Net Income"].head(4)
                    result["quarterly_net_income"] = {
                        str(k)[:10]: float(v) if pd.notnull(v) else 0.0
                        for k, v in inc_series.items()
                    }
                else:
                    result["quarterly_net_income"] = {}
        except Exception:
            pass

        _fund_cache[key] = result
        return result
    except Exception as exc:
        logger.error("Fundamentals fetch failed for %s: %s", symbol, exc)
        return {
            "symbol": symbol,
            "exchange": exchange,
            "error": str(exc),
            "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
        }


def get_nifty50_data() -> Dict[str, Any]:
    """Fetch latest Nifty 50 data for market regime assessment."""
    return get_live_price("^NSEI", "NSE")


def get_india_vix() -> Dict[str, Any]:
    """Fetch India VIX (^INDIAVIX)."""
    key = "vix"
    if key in _price_cache:
        return _price_cache[key]
    try:
        tk = yf.Ticker("^INDIAVIX")
        info = tk.fast_info
        vix = float(info.last_price or 0)
        result = {
            "vix": vix,
            "regime": "high" if vix > 20 else "normal" if vix > 13 else "low",
            "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
        }
        _price_cache[key] = result
        return result
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return {"vix": None, "regime": "unknown", "source": "yfinance",
                "timestamp": datetime.utcnow().isoformat(), "error": str(exc)}


def get_top_movers(n: int = 10) -> List[Dict[str, Any]]:
    """Return top gainers/losers from a predefined Nifty 50 list."""
    NIFTY50 = [
        "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
        "INFY", "SBIN", "HINDUNILVR", "ITC", "LT",
        "KOTAKBANK", "AXISBANK", "WIPRO", "HCLTECH", "MARUTI",
        "SUNPHARMA", "BAJFINANCE", "TITAN", "NESTLEIND", "NTPC",
    ]
    results = []
    for sym in NIFTY50:
        try:
            data = get_live_price(sym)
            results.append(data)
        except Exception:
            continue
    results.sort(key=lambda x: x.get("change_pct", 0), reverse=True)
    return results[:n]
