"""
agents/state.py – LangGraph state definition shared across all agent nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Shared state that flows through the LangGraph pipeline.
    All fields are optional – nodes populate them as they run.
    """
    # ── Query ──────────────────────────────────────────────────────────────────
    user_query: str = ""
    symbol: str = ""
    exchange: str = "NSE"
    image_path: Optional[str] = None      # path to uploaded image if any
    session_id: str = ""

    # ── Data outputs from specialist agents ───────────────────────────────────
    price_data: Dict[str, Any] = field(default_factory=dict)
    fundamentals: Dict[str, Any] = field(default_factory=dict)
    news_items: List[Dict[str, Any]] = field(default_factory=list)
    news_sentiment: Dict[str, Any] = field(default_factory=dict)
    fii_data: Dict[str, Any] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)    # nifty + vix
    mf_data: Dict[str, Any] = field(default_factory=dict)
    fd_data: Dict[str, Any] = field(default_factory=dict)
    vision_analysis: Dict[str, Any] = field(default_factory=dict)

    # ── Technical analysis ────────────────────────────────────────────────────
    trade_setup: Optional[Any] = None       # TradeSetup object
    backtest_result: Optional[Any] = None   # BacktestResult object

    # ── Confidence engine ─────────────────────────────────────────────────────
    confidence_report: Optional[Any] = None  # ConfidenceReport object
    past_signal_stats: Dict[str, Any] = field(default_factory=dict)

    # ── Risk manager ──────────────────────────────────────────────────────────
    position_size: float = 0.0
    capital: float = 100_000.0

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: str = ""
    route: str = "analyze"    # "analyze" | "mf" | "fd" | "watchlist" | "portfolio"
    error: Optional[str] = None
