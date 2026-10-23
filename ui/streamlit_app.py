"""
ui/streamlit_app.py – Main Streamlit frontend.
Run with: streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_core.messages import HumanMessage

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Investment Assistant – India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after page config) ───────────────────────────────────────────────
from agents.graph import get_graph
from agents.state import AgentState
from db.database import init_db
from tools.equity_data import get_live_price, get_top_movers
from ui.charts import build_candlestick_chart, build_confidence_gauge, build_breakdown_bar
from strategies.indicators import add_all_indicators
from tools.equity_data import get_historical_data

# ── Async helper ─────────────────────────────────────────────────────────────
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background: #0f0f23; }
.stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }

.chat-bubble-user {
    background: linear-gradient(135deg, #1e3a5f, #0d47a1);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e3f2fd;
    margin-left: 20%;
}
.chat-bubble-assistant {
    background: linear-gradient(135deg, #1a2744, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e0e0e0;
    margin-right: 5%;
}

.metric-card {
    background: linear-gradient(135deg, #1a2744, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.pass-badge {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: #a5d6a7;
    border-radius: 8px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 0.85rem;
}
.fail-badge {
    background: linear-gradient(135deg, #b71c1c, #c62828);
    color: #ffcdd2;
    border-radius: 8px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 0.85rem;
}
.stButton>button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #0f0f23 100%);
    border-right: 1px solid #1e3a5f;
}
.disclaimer {
    background: rgba(183, 28, 28, 0.1);
    border: 1px solid #b71c1c;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.75rem;
    color: #ef9a9a;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialization ──────────────────────────────────────────────
def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "capital" not in st.session_state:
        from core.config import get_settings
        st.session_state.capital = float(get_settings().default_capital)
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Trading")
    st.markdown("*Indian Market Intelligence*")
    st.divider()

    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Top Movers", use_container_width=True):
            st.session_state.quick_action = "top_movers"
        if st.button("🏦 Best FD Now", use_container_width=True):
            st.session_state.quick_action = "best_fd"
    with col2:
        if st.button("🔍 Scan 75%+", use_container_width=True):
            st.session_state.quick_action = "scan"
        if st.button("📂 Compare MFs", use_container_width=True):
            st.session_state.quick_action = "compare_mf"

    st.divider()

    # User profile
    st.markdown("### 👤 Your Profile")
    capital = st.number_input("Capital (₹)", value=st.session_state.capital,
                               min_value=100.0, step=100.0, format="%.0f")
    st.session_state.capital = capital

    risk = st.select_slider("Risk Appetite",
                             options=["Conservative", "Moderate", "Aggressive"],
                             value="Moderate")
    horizon = st.selectbox("Trade Horizon",
                           ["Intraday", "Swing (2-10 days)", "Positional (weeks)"])

    st.divider()

    # Watchlist
    st.markdown("### 👁️ Watchlist")
    new_watch = st.text_input("Add symbol (e.g. RELIANCE)")
    if st.button("➕ Add") and new_watch:
        if new_watch.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_watch.upper())
    for sym in st.session_state.watchlist:
        try:
            data = get_live_price(sym)
            chg = data.get("change_pct", 0)
            color = "🟢" if chg >= 0 else "🔴"
            st.caption(f"{color} {sym}: ₹{data.get('price', 0):.1f} ({chg:+.1f}%)")
        except Exception:
            st.caption(f"⚪ {sym}")

    st.divider()
    st.markdown('<div class="disclaimer">⚠️ Educational analysis only. Not SEBI-registered advice. Past performance ≠ future results.</div>',
                unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🤖 AI Investment Assistant")
st.markdown("*Analyze Indian equities, mutual funds, FDs & bonds with AI-powered insights*")

tabs = st.tabs(["💬 Chat", "📊 Chart & Analysis", "📋 Signal Report", "📈 Backtest", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f'<div class="chat-bubble-user">👤 {content}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-assistant">{content}</div>',
                            unsafe_allow_html=True)

    # Image upload + input
    col_img, col_inp = st.columns([1, 5])
    with col_img:
        uploaded = st.file_uploader("📷", type=["png","jpg","jpeg","webp"],
                                    label_visibility="collapsed")
    with col_inp:
        user_input = st.chat_input("Ask anything... e.g. 'Analyze RELIANCE for swing trade'")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Save uploaded image temporarily
        img_path = None
        if uploaded:
            img_dir = Path("data/uploads")
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = str(img_dir / f"{uuid.uuid4()}_{uploaded.name}")
            with open(img_path, "wb") as f:
                f.write(uploaded.read())

        # Build initial state
        langchain_messages = [
            HumanMessage(content=m["content"])
            for m in st.session_state.messages
            if m["role"] == "user"
        ]

        initial_state = {
            "messages": langchain_messages,
            "user_query": user_input,
            "session_id": st.session_state.session_id,
            "image_path": img_path,
            "capital": st.session_state.capital,
        }

        with st.spinner("🧠 Analyzing... (this may take 15–30s for full backtest)"):
            try:
                graph = get_graph()
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                final_state = run_async(
                    graph.ainvoke(initial_state, config=config)
                )
                st.session_state.last_state = final_state
                response = final_state.get("final_response", "I encountered an error. Please try again.")
            except Exception as exc:
                response = f"⚠️ Error: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: CHART & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        chart_symbol = st.text_input("Symbol", value="RELIANCE", key="chart_sym")
    with col2:
        chart_period = st.selectbox("Period", ["1y", "2y", "5y", "6mo", "3mo"], index=0)
    with col3:
        chart_interval = st.selectbox("Interval", ["1d", "1wk", "1h"], index=0)

    if st.button("📊 Load Chart"):
        with st.spinner("Loading chart..."):
            df = get_historical_data(chart_symbol, "NSE", chart_period, chart_interval)
            if not df.empty:
                from core.config import get_settings
                cfg = get_settings().yaml_section("strategies")
                df = add_all_indicators(df, cfg)
                setup = st.session_state.last_state.get("trade_setup") if st.session_state.last_state else None
                fig = build_candlestick_chart(df, chart_symbol, setup=setup)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"No data found for {chart_symbol}")

    # Live price cards
    if st.button("🔄 Refresh Top Movers"):
        movers = get_top_movers(10)
        cols = st.columns(5)
        for i, m in enumerate(movers):
            with cols[i % 5]:
                chg = m.get("change_pct", 0)
                color = "🟢" if chg >= 0 else "🔴"
                st.metric(
                    label=m["symbol"],
                    value=f"₹{m.get('price', 0):.1f}",
                    delta=f"{chg:+.2f}%",
                )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: SIGNAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    last = st.session_state.last_state
    if last is None:
        st.info("💡 Run an equity analysis in the Chat tab to see the signal report here.")
    else:
        report = last.get("confidence_report")
        setup = last.get("trade_setup")
        symbol = last.get("symbol", "")

        if report:
            # Confidence gauge + gate status
            col_gauge, col_status = st.columns([2, 1])
            with col_gauge:
                st.plotly_chart(build_confidence_gauge(report.score), use_container_width=True)
            with col_status:
                st.markdown("### Gate Status")
                if report.passed_gate():
                    st.markdown('<span class="pass-badge">✅ PASS – Trade recommended</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<span class="fail-badge">❌ FAIL – NO TRADE</span>',
                                unsafe_allow_html=True)
                if report.hard_rule_failures:
                    st.markdown("**Hard rule failures:**")
                    for f in report.hard_rule_failures:
                        st.markdown(f"- {f}")

            # Breakdown chart
            st.plotly_chart(build_breakdown_bar(report.breakdown), use_container_width=True)

        if setup and setup.direction != "NO_SIGNAL":
            st.markdown("### 📋 Trade Setup")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Direction", setup.direction)
            col2.metric("Entry", f"₹{setup.entry_price:.2f}")
            col3.metric("Stop-Loss", f"₹{setup.stop_loss:.2f}")
            col4.metric("Risk:Reward", f"1:{setup.risk_reward:.1f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Target 1", f"₹{setup.target1:.2f}")
            col6.metric("Target 2", f"₹{setup.target2:.2f}")
            col7.metric("ATR", f"₹{setup.atr:.2f}")
            col8.metric("Position Size", f"{last.get('position_size', 0):.0f} shares")

            st.markdown("**Strategies triggered:**")
            for s in setup.strategies_triggered:
                st.success(f"✅ {s}")
            if setup.warnings:
                for w in setup.warnings:
                    st.warning(w)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    last = st.session_state.last_state
    if last is None:
        st.info("💡 Run an equity analysis in the Chat tab to see backtest results here.")
    else:
        bt = last.get("backtest_result")
        if bt:
            st.markdown(f"### 📊 Backtest: {bt.symbol} – {bt.strategy_name}")
            if bt.warnings:
                for w in bt.warnings:
                    st.warning(w)

            cols = st.columns(4)
            cols[0].metric("Total Trades", bt.total_trades,
                           delta="✅ Valid" if bt.is_valid else "❌ Too few")
            cols[1].metric("Win Rate", f"{bt.win_rate}%")
            cols[2].metric("Profit Factor", f"{bt.profit_factor:.2f}")
            cols[3].metric("Expectancy", f"{bt.expectancy:.2f}%")

            cols2 = st.columns(4)
            cols2[0].metric("CAGR", f"{bt.cagr:.1f}%")
            cols2[1].metric("Max Drawdown", f"{bt.max_drawdown:.1f}%")
            cols2[2].metric("Sharpe Ratio", f"{bt.sharpe:.2f}")
            cols2[3].metric("Buy & Hold", f"{bt.buy_hold_return_pct:.1f}%")

            oos_label = "✅ Valid" if bt.is_out_of_sample_valid else "❌ Negative expectancy"
            st.info(f"**Out-of-sample test:** {oos_label}")
        else:
            st.info("No backtest data available.")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
## 🤖 AI Investment Assistant – India

Built with **LangChain + LangGraph** multi-agent architecture.

### 🏗️ Agent Pipeline
```
User Query
    ↓
Supervisor (routes to specialist)
    ↓
Vision → Technical + Fundamental + News (parallel)
    ↓
Backtest & Validation
    ↓
Confidence Engine (deterministic, weighted scoring)
    ↓
HARD GATE (score > 75% required)
    ├── PASS → Risk Manager → Synthesizer → BUY/SELL with full details
    └── FAIL → NO TRADE (with explanation)
```

### ⚙️ Tech Stack
- **LLM**: Groq `openai/gpt-oss-120b` (fast high-reasoning LLM)
- **Framework**: LangChain + LangGraph (multi-agent, stateful)
- **Data**: yfinance, Angel One SmartAPI, mfapi.in, NSE RSS feeds
- **Indicators**: Pure-pandas & numpy (EMA, RSI, MACD, Supertrend, ADX, BB, ATR)
- **Backtesting**: Walk-forward (70/30 split)
- **Frontend**: Streamlit
- **Database**: SQLite (async via aiosqlite)

### 📊 Confidence Score Weights
| Factor | Weight |
|--------|--------|
| Indicator confluence | 25% |
| Multi-timeframe alignment | 15% |
| Backtest edge | 25% |
| Market regime (Nifty/VIX/FII) | 10% |
| Liquidity quality | 5% |
| Fundamentals & news | 10% |
| Agent track record | 10% |

### ⚠️ Disclaimer
This tool provides **educational analysis only** and is NOT SEBI-registered investment advice.
All trading involves risk. A 75%+ confidence score is a statistical estimate, not a guarantee.
Past performance does not guarantee future results. Always consult a SEBI-registered advisor.
""")
