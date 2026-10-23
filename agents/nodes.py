"""
agents/nodes.py – All LangGraph node functions.
Each node takes AgentState, performs work, and returns state updates.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.llm_provider import get_llm
from agents.prompts import (
    EQUITY_RESEARCH_ANALYST_PROMPT,
    EXPLAINER_RISK_REVIEWER_SYSTEM_PROMPT,
    QUANT_INTRADAY_SYSTEM_PROMPT,
    TECHNICAL_ANALYSIS_SPECIALIST_PROMPT,
)
from agents.state import AgentState
from backtester.engine import run_backtest
from confidence.engine import compute_confidence
from strategies.setup_engine import analyze_setup
from tools.equity_data import (
    get_fundamentals,
    get_historical_data,
    get_india_vix,
    get_live_price,
    get_nifty50_data,
)
from tools.fixed_income import get_fd_rates, get_best_fd, get_rbi_rates
from tools.mf_data import compute_returns, search_scheme
from tools.news_data import fetch_news, get_fii_dii_data, simple_sentiment_score

logger = logging.getLogger(__name__)


# ── 1. Supervisor / Router ────────────────────────────────────────────────────
async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Route the query to the appropriate specialist agent(s).
    Returns: route + extracted symbol.
    """
    llm = get_llm(streaming=False)
    query = state.get("user_query", "")

    prompt = f"""You are a query router for an Indian investment assistant.
Analyze the user query and respond with EXACTLY ONE of these routes and optionally a stock symbol.

Routes:
- "analyze" – for stock/equity technical or fundamental analysis
- "mf"      – for mutual fund queries
- "fd"      – for fixed deposit or bond queries
- "news"    – for market news or sentiment only
- "portfolio" – for portfolio review
- "watchlist" – for watchlist management
- "chat"    – for general financial questions

Respond in this exact JSON format (no explanation):
{{"route": "<route>", "symbol": "<SYMBOL_OR_EMPTY>", "exchange": "NSE"}}

User query: {query}
"""
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        import json, re
        match = re.search(r'\{.*\}', resp.content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return {
                "route": parsed.get("route", "analyze"),
                "symbol": parsed.get("symbol", "").upper().strip(),
                "exchange": parsed.get("exchange", "NSE"),
            }
    except Exception as exc:
        logger.error("Supervisor routing error: %s", exc)

    return {"route": "chat", "symbol": "", "exchange": "NSE"}


# ── 2. Vision Agent ───────────────────────────────────────────────────────────
async def vision_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze uploaded chart/portfolio image using Groq vision.
    Extracts: symbol, timeframe, key levels, patterns, holdings.
    """
    image_path = state.get("image_path")
    if not image_path or not Path(image_path).exists():
        return {"vision_analysis": {}}

    llm = get_llm(streaming=False)
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        suffix = Path(image_path).suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(suffix[1:], "image/png")

        messages = [
            SystemMessage(content=(
                "You are a chart analysis expert for Indian equity markets. "
                "Extract all visible information from this trading chart or portfolio image."
            )),
            HumanMessage(content=[
                {"type": "text", "text": (
                    "Analyze this image. Identify: "
                    "1. Stock symbol and exchange (NSE/BSE) if visible "
                    "2. Timeframe (1m/5m/15m/1h/1d/1w) "
                    "3. Key support/resistance levels "
                    "4. Visible chart patterns (head & shoulders, triangle, etc.) "
                    "5. Indicator values visible (RSI, MACD, etc.) "
                    "6. Holdings if it's a portfolio image "
                    "Return as structured JSON."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{img_b64}"
                }},
            ]),
        ]
        resp = await llm.ainvoke(messages)
        import json, re
        match = re.search(r'\{.*\}', resp.content, re.DOTALL)
        if match:
            analysis = json.loads(match.group())
        else:
            analysis = {"raw_analysis": resp.content}

        # If vision found a symbol, use it
        extracted_symbol = analysis.get("symbol", "")
        updates: Dict[str, Any] = {"vision_analysis": analysis}
        if extracted_symbol and not state.get("symbol"):
            updates["symbol"] = extracted_symbol.upper()
        return updates

    except Exception as exc:
        logger.error("Vision analysis failed: %s", exc)
        return {"vision_analysis": {"error": str(exc)}}


# ── 3. Equity Technical Agent ─────────────────────────────────────────────────
async def technical_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetch historical data, compute indicators, generate trade setup.
    """
    symbol = state.get("symbol", "")
    exchange = state.get("exchange", "NSE")
    if not symbol:
        return {"trade_setup": None, "price_data": {}}

    # Live price
    price_data = get_live_price(symbol, exchange)

    # Historical data (daily + weekly)
    df_daily = await asyncio.to_thread(get_historical_data, symbol, exchange, "5y", "1d")
    df_weekly = await asyncio.to_thread(get_historical_data, symbol, exchange, "5y", "1wk")

    # Trade setup
    setup = analyze_setup(
        symbol=symbol,
        df_daily=df_daily,
        df_weekly=df_weekly if not df_weekly.empty else None,
        current_price=price_data.get("price"),
    )

    return {"price_data": price_data, "trade_setup": setup}


# ── 4. Equity Fundamental Agent ───────────────────────────────────────────────
async def fundamental_agent_node(state: AgentState) -> Dict[str, Any]:
    """Fetch PE, ROE, D/E, quarterly results."""
    symbol = state.get("symbol", "")
    exchange = state.get("exchange", "NSE")
    if not symbol:
        return {"fundamentals": {}}
    fundamentals = await asyncio.to_thread(get_fundamentals, symbol, exchange)
    return {"fundamentals": fundamentals}


# ── 5. News & Sentiment Agent ─────────────────────────────────────────────────
async def news_agent_node(state: AgentState) -> Dict[str, Any]:
    """Fetch news + sentiment + FII/DII + market regime."""
    symbol = state.get("symbol")
    news = await asyncio.to_thread(fetch_news, symbol, 20)
    sentiment = simple_sentiment_score([
        f"{n.get('title','')} {n.get('summary','')}" for n in news
    ])
    fii = await asyncio.to_thread(get_fii_dii_data)
    nifty = await asyncio.to_thread(get_nifty50_data)
    vix = await asyncio.to_thread(get_india_vix)

    return {
        "news_items": news,
        "news_sentiment": sentiment,
        "fii_data": fii,
        "market_data": {"nifty": nifty, "vix": vix},
    }


# ── 6. Mutual Fund Agent ──────────────────────────────────────────────────────
async def mf_agent_node(state: AgentState) -> Dict[str, Any]:
    """Search MF schemes and compute returns."""
    query = state.get("user_query", "")
    schemes = await asyncio.to_thread(search_scheme, query)
    results = []
    for s in schemes[:5]:
        code = s.get("schemeCode")
        if code:
            ret = await asyncio.to_thread(compute_returns, code)
            ret["scheme_name"] = s.get("schemeName")
            results.append(ret)
    return {"mf_data": {"schemes": results}}


# ── 7. Fixed Income Agent ─────────────────────────────────────────────────────
async def fd_agent_node(state: AgentState) -> Dict[str, Any]:
    """Fetch FD rates, G-Sec yields, RBI rates."""
    fd = get_fd_rates()
    rbi = get_rbi_rates()
    return {"fd_data": {"fd_rates": fd, "rbi": rbi}}


# ── 8. Backtest Node ──────────────────────────────────────────────────────────
async def backtest_node(state: AgentState) -> Dict[str, Any]:
    """Run walk-forward backtest for the proposed setup."""
    symbol = state.get("symbol", "")
    exchange = state.get("exchange", "NSE")
    if not symbol:
        from backtester.engine import BacktestResult
        return {"backtest_result": BacktestResult(symbol="", strategy_name="")}

    df_daily = await asyncio.to_thread(get_historical_data, symbol, exchange, "5y", "1d")
    from core.config import get_settings
    settings = get_settings()
    cfg = settings.yaml_section("strategies")
    bt_cfg = settings.yaml_section("backtest")

    result = await asyncio.to_thread(
        run_backtest,
        symbol, df_daily,
        commission_pct=bt_cfg.get("commission_pct", 0.03),
        slippage_pct=bt_cfg.get("slippage_pct", 0.05),
        cfg=cfg,
    )
    return {"backtest_result": result}


# ── 9. Confidence Engine Node ─────────────────────────────────────────────────
async def confidence_node(state: AgentState) -> Dict[str, Any]:
    """Compute deterministic confidence score."""
    setup = state.get("trade_setup")
    bt = state.get("backtest_result")
    if setup is None or bt is None:
        return {"confidence_report": None}

    from backtester.engine import BacktestResult
    from strategies.setup_engine import TradeSetup
    from confidence.engine import ConfidenceReport

    report = compute_confidence(
        setup=setup,
        bt=bt,
        market_data=state.get("market_data", {}),
        fundamentals=state.get("fundamentals", {}),
        news_sentiment=state.get("news_sentiment", {}),
        past_signal_stats=state.get("past_signal_stats", {}),
        price_data=state.get("price_data", {}),
        fii_data=state.get("fii_data", {}),
    )
    return {"confidence_report": report}


# ── 10. Risk Manager Node ─────────────────────────────────────────────────────
async def risk_manager_node(state: AgentState) -> Dict[str, Any]:
    """Compute position size based on ATR stop, 1% capital risk, 20% max position value, and India VIX."""
    setup = state.get("trade_setup")
    capital = state.get("capital", 100_000.0)
    from core.config import get_settings
    risk_pct = get_settings().risk_per_trade_pct / 100

    if setup is None or setup.entry_price == 0 or setup.stop_loss == 0 or setup.direction == "NO_SIGNAL":
        return {"position_size": 0.0}

    risk_per_share = abs(setup.entry_price - setup.stop_loss)
    if risk_per_share == 0:
        return {"position_size": 0.0}

    capital_at_risk = capital * risk_pct
    shares = int(capital_at_risk / risk_per_share)

    # 1. Cap position value at max 20% of capital
    max_value_shares = int((capital * 0.20) / setup.entry_price)
    shares = min(shares, max_value_shares)

    # 2. Volatility bands (India VIX scaling)
    # <13 low, 13-18 normal, 18-24 elevated (reduce by 50%), >24 high (no new positions)
    vix_dict = state.get("market_data", {}).get("vix", {})
    vix = vix_dict.get("price") or vix_dict.get("vix")
    if vix is not None:
        try:
            vix_val = float(vix)
            if vix_val > 24.0:
                shares = 0
            elif vix_val >= 18.0:
                shares = int(shares * 0.5)
        except (ValueError, TypeError):
            pass

    return {"position_size": float(shares)}


# ── 11. Synthesizer Node ──────────────────────────────────────────────────────
async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate the final human-readable response.
    The LLM explains the pre-computed score – it never changes the numbers.
    """
    llm = get_llm(streaming=True)
    report = state.get("confidence_report")
    setup = state.get("trade_setup")
    bt = state.get("backtest_result")
    news = state.get("news_items", [])[:5]
    fundamentals = state.get("fundamentals", {})
    route = state.get("route", "chat")
    fundamentals = state.get("fundamentals", {})
    market_data = state.get("market_data", {})
    fii_data = state.get("fii_data", {})
    position_size = state.get("position_size", 0.0)
    capital = state.get("capital", 100_000.0)

    # Build context block
    ctx = f"""
Route: {route}
Symbol: {state.get('symbol', 'N/A')} ({state.get('exchange', 'NSE')})
Current price: ₹{state.get('price_data', {}).get('price', 'N/A')}
Portfolio Capital: ₹{capital:,.2f}
Calculated Position Size: {position_size:.0f} shares
Vision analysis: {state.get('vision_analysis', {})}
"""
    if setup:
        ctx += f"""
Trade Setup:
  Direction: {setup.direction}
  Timeframe: {setup.timeframe}
  Detected Regime: {setup.regime}
  Eligible Instrument: {setup.eligible_instrument}
  Signal Families Triggered: {', '.join(setup.families_triggered) if setup.families_triggered else 'None'}
  Entry: ₹{setup.entry_price:.2f}
  Stop-Loss: ₹{setup.stop_loss:.2f}
  Target 1: ₹{setup.target1:.2f}
  Target 2: ₹{setup.target2:.2f}
  Risk:Reward: 1:{setup.risk_reward:.2f}
  ATR: ₹{setup.atr:.2f}
  Strategies triggered: {', '.join(setup.strategies_triggered)}
  Candlestick patterns: {', '.join(setup.candlestick_patterns)}
  Warnings: {'; '.join(setup.warnings)}
"""
    if report:
        calib_txt = "Calibrated" if report.is_calibrated else "Uncalibrated (requires ≥50 logged signals)"
        ctx += f"""
Confidence Score: {report.score:.1f}% ({'PASS ✅' if report.passed_gate() else 'FAIL ❌ (HARD GATE)'})
Calibration Status: {calib_txt}
Breakdown: {report.breakdown}
Hard rule failures: {report.hard_rule_failures}
Low scoring factors: {report.reasons_low}
High scoring factors: {report.reasons_high}
Report Timestamp: {report.timestamp}
"""
    if bt:
        ctx += f"""
Backtest ({bt.strategy_name}):
  Trades: {bt.total_trades}, Win rate: {bt.win_rate:.1f}%
  Profit factor: {bt.profit_factor:.2f}, Expectancy: {bt.expectancy:.2f}%
  CAGR: {bt.cagr:.1f}%, Max DD: {bt.max_drawdown:.1f}%, Sharpe: {bt.sharpe:.2f}
  Buy & Hold: {bt.buy_hold_return_pct:.1f}%
  Out-of-sample valid: {bt.is_out_of_sample_valid}
  Warnings: {'; '.join(bt.warnings)}
"""
    if market_data:
        nifty = market_data.get("nifty", {})
        vix = market_data.get("vix", {})
        ctx += f"""
Market Regime Context:
  Nifty 50: ₹{nifty.get('price', 'N/A')} ({nifty.get('change_pct', 0):+.2f}%)
  India VIX: {vix.get('price', 'N/A')} ({vix.get('regime', 'Normal')})
  FII/DII Net Flow: {fii_data.get('summary', 'Unavailable')}
"""
    if fundamentals:
        ctx += f"""
Fundamentals:
  PE: {fundamentals.get('pe', 'N/A')}, Forward PE: {fundamentals.get('forward_pe', 'N/A')}
  ROE: {fundamentals.get('roe', 'N/A')}, ROCE: {fundamentals.get('roce', 'N/A')}
  Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}
  Market Cap: ₹{fundamentals.get('market_cap', 0) / 1e7:,.1f} Cr
  52-Week High: ₹{fundamentals.get('fifty_two_week_high', 'N/A')}, Low: ₹{fundamentals.get('fifty_two_week_low', 'N/A')}
  Quarterly Revenue Growth: {fundamentals.get('revenue_growth_qoq', 'N/A')}%
"""
    if news:
        ctx += "\nTop news:\n" + "\n".join(f"- {n.get('title','')} ({n.get('source','')})" for n in news[:8])

    from confidence.engine import THRESHOLD

    query_lower = state.get("user_query", "").lower()
    is_fundamental_query = any(k in query_lower for k in ["invest", "fundamental", "long-term", "financial", "growth", "balance sheet", "research", "valuation", "ratio"])

    if is_fundamental_query:
        base_prompt = EQUITY_RESEARCH_ANALYST_PROMPT
    else:
        base_prompt = EXPLAINER_RISK_REVIEWER_SYSTEM_PROMPT

    system_prompt = f"""{base_prompt}

# ENGINE DETERMINISTIC DATA VERIFICATION:
- You must report exactly what the deterministic engine calculated below. Do NOT compute, alter, or round differently.
- Hard Threshold: {THRESHOLD}%.
- If report.passed_gate() is False or setup.direction is NO_SIGNAL, your Decision in section 1 MUST be NO-TRADE (or WATCH).
- Strictly adhere to the 6-part OUTPUT FORMAT:
  1) Decision: TRADE / NO-TRADE / WATCH
  2) Regime summary (with data timestamps and sources)
  3) Setup: signal families that agree, instrument, entry, stop, T1, T2, size (shares or lots), R:R after costs
  4) Pillar scores and total score as reported by the engine
  5) Risks and what would invalidate the setup
  6) Data gaps or assumptions
- Conclude with: 'Not investment advice. Paper-trade first.'"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User query: {state.get('user_query', '')}\n\nDeterministic Engine Output:\n{ctx}"),
    ]

    # Streaming synthesis
    response_text = ""
    try:
        async for chunk in llm.astream(messages):
            response_text += chunk.content
    except Exception as exc:
        logger.error("Synthesizer streaming failed: %s", exc)
        response = await llm.ainvoke(messages)
        response_text = response.content

    # Force NO TRADE if gate failed
    if report and not report.passed_gate():
        if "NO-TRADE" not in response_text[:200].upper() and "NO TRADE" not in response_text[:200].upper():
            response_text = (
                f"### 1) Decision: NO-TRADE\n\n"
                f"**Confidence Gate: {report.score}% (Threshold: >{THRESHOLD}%)**\n\n"
                f"**Hard Rule Failures:** {'; '.join(report.hard_rule_failures) if report.hard_rule_failures else 'Confidence score below threshold'}\n\n"
                + response_text
            )

    if "Not investment advice. Paper-trade first." not in response_text:
        response_text += "\n\n---\n*Not investment advice. Paper-trade first.*"

    return {
        "final_response": response_text,
        "messages": [AIMessage(content=response_text)],
    }


# ── No Trade Node ─────────────────────────────────────────────────────────────
async def no_trade_node(state: AgentState) -> Dict[str, Any]:
    """Called when the confidence gate fails – generates a clear NO TRADE response."""
    from confidence.engine import THRESHOLD
    report = state.get("confidence_report")
    setup = state.get("trade_setup")
    score = report.score if report else 0
    failures = report.hard_rule_failures if report else ["Score below threshold"]
    low = report.reasons_low if report else []

    symbol = state.get("symbol", "this asset")
    msg = f"""## 🚫 NO TRADE – {symbol}

**Confidence Score: {score}% (Required: strictly > {THRESHOLD}%)**

### Why this trade was rejected:
{chr(10).join(f'- {f}' for f in failures)}

### Lowest scoring factors:
{chr(10).join(f'- {r}' for r in low)}

### What would need to improve:
- Indicator confluence must be stronger (multiple strategies confirming)
- Backtest must have ≥30 trades with positive out-of-sample expectancy
- Risk:Reward must be ≥1:2
- No conflicting higher-timeframe trend

_Would you like to add {symbol} to your watchlist? I'll alert you when confidence crosses {THRESHOLD}%._

---
⚠️ **Disclaimer:** This is educational analysis, not SEBI-registered investment advice. Markets carry risk.
"""
    return {
        "final_response": msg,
        "messages": [AIMessage(content=msg)],
    }


# ── General Chat Node ─────────────────────────────────────────────────────────
async def chat_node(state: AgentState) -> Dict[str, Any]:
    """Handle general financial Q&A without trade signals."""
    llm = get_llm(streaming=True)
    messages = state.get("messages", [])
    system = SystemMessage(content=(
        "You are a knowledgeable Indian investment assistant. "
        "Answer questions about investing, markets, tax, financial planning for Indian context. "
        "Always add the SEBI disclaimer at the end. Never guarantee returns."
    ))
    response_text = ""
    async for chunk in llm.astream([system] + messages):
        response_text += chunk.content

    return {
        "final_response": response_text,
        "messages": [AIMessage(content=response_text)],
    }
