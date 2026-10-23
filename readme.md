# 🤖 AI Investment Assistant – India

A **production-quality agentic AI investment assistant** for the Indian market built with **LangChain + LangGraph**, powered by **Groq (Llama 4 Scout)** with full vision capabilities.

> ⚠️ **Disclaimer**: Educational analysis only. Not SEBI-registered investment advice. Past performance ≠ future results. All trading involves risk.

---

## 🏗️ Architecture

```
User Query / Chart Image
         │
    ┌────▼─────┐
    │ Supervisor│  ← Routes to specialist agents
    └────┬─────┘
         │
    ┌────▼──────┐   ┌──────────────────┐   ┌───────────────┐
    │  Vision   │   │ Technical Agent  │   │Fundamental    │
    │  Agent    │──►│ (EMA/RSI/MACD/   │   │Agent (PE/ROE/ │
    │ (Llama 4) │   │  Supertrend/ADX) │   │ D/E/growth)   │
    └───────────┘   └────────┬─────────┘   └───────┬───────┘
                             │                     │
                    ┌────────▼─────────────────────▼───────┐
                    │       News & Sentiment Agent          │
                    │  (RSS feeds, FII/DII, Nifty, VIX)    │
                    └────────────────────┬──────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Backtest/Validation │
                              │  (Walk-forward 70/30)│
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Confidence Engine  │
                              │  (Deterministic,    │
                              │   Weighted 0–100)   │
                              └──────────┬───────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │        HARD GATE             │
                          │   Score strictly > 90%?      │
                          └──────┬──────────────┬────────┘
                               YES              NO
                          ┌─────▼──────┐   ┌───▼────────┐
                          │Risk Manager│   │  NO TRADE  │
                          └─────┬──────┘   └────────────┘
                                │
                         ┌──────▼──────┐
                         │ Synthesizer │  ← LLM explains pre-computed score
                         └─────────────┘
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **9 Specialist Agents** | Technical, Fundamental, News, MF, FD/Bond, Backtest, Confidence, Risk Manager, Vision |
| **90% Hard Gate** | BUY/SELL only when deterministic score > 90% AND all hard rules pass |
| **Vision Agent** | Upload chart/portfolio screenshots — Llama 4 Scout extracts symbols, levels, patterns |
| **Full Backtest** | 5Y walk-forward (70/30 split), win rate, profit factor, Sharpe, max drawdown |
| **7 Strategies** | EMA trend, RSI, MACD, ADX, Supertrend, Donchian breakout, Bollinger reversion |
| **MF Analysis** | NAV, 1Y/3Y/5Y returns, SIP calculator via mfapi.in |
| **FD/Bond** | Rates for SBI/HDFC/ICICI/Axis/Yes/Post Office, G-Sec yields, real returns |
| **Plotly Charts** | Interactive candlestick + indicators + trade setup lines |
| **Confidence Gauge** | Live red/green gauge with factor breakdown bar chart |
| **Conversation Memory** | LangGraph MemorySaver checkpointer + SQLite persistence |

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq `meta-llama/llama-4-scout-17b-16e-instruct` (free, vision, 128K ctx) |
| Agent Framework | LangChain + LangGraph (multi-agent, stateful) |
| Backend | Python 3.11+, FastAPI-ready |
| Frontend | Streamlit (dark theme, glassmorphism UI) |
| Market Data | yfinance (.NS/.BO), mfapi.in, NSE RSS feeds |
| Indicators | pandas-ta (EMA, RSI, MACD, Supertrend, ADX, ATR, BB, Donchian) |
| Backtesting | Manual walk-forward (vectorbt optional) |
| Charts | Plotly (interactive, dark theme) |
| Database | SQLite + SQLAlchemy async (aiosqlite) |
| Cache | cachetools TTLCache (1min live, 24h fundamentals) |

---

## 🚀 Quick Start

### 1. Clone & Set Up

```powershell
cd AI_Trading
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in:
```env
GROQ_API_KEY=your_groq_api_key_here   # Get free at console.groq.com
```

Angel One SmartAPI (optional, for live order routing):
```env
ANGEL_API_KEY=your_key
ANGEL_CLIENT_ID=your_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_base32_secret
```

### 3. Run

```powershell
# Option A – launch everything (DB init + Streamlit)
python main.py

# Option B – Streamlit directly
streamlit run ui/streamlit_app.py

# Option C – run tests
pytest tests/ -v
```

App opens at **http://localhost:8501**

---

## 📊 Confidence Score Weights

```yaml
indicator_confluence:  25%   # EMA/RSI/MACD/Supertrend/ADX/BB agree?
multi_timeframe:       15%   # Weekly + daily + intraday alignment
backtest_edge:         25%   # Win rate, profit factor, OOS expectancy
market_regime:         10%   # Nifty trend, India VIX, FII/DII flows
liquidity_quality:      5%   # Market cap, volume (no penny stocks)
fundamentals_news:     10%   # PE ratio, ROE, news sentiment
agent_track_record:    10%   # Past signal hit rate from DB
```

**Hard Rules (any failure = automatic NO TRADE):**
1. Score must be **strictly > 90** (90.0 itself is rejected)
2. Risk:Reward must be **≥ 1:2**
3. Backtest must have **≥ 30 trades** with positive out-of-sample expectancy
4. Price data must be **< 5 min old** during market hours
5. No conflicting higher-timeframe trend
6. No major event risk (results, ex-date, RBI policy) in the trade window

---

## 💬 Example Prompts

```
✅ "Analyze RELIANCE NSE for a swing trade"
✅ "Upload a chart image and ask: what's the trade setup?"
✅ "Compare top 3 ELSS mutual funds"
✅ "Best FD rates for 1 year, ₹5 lakhs, senior citizen"
✅ "Backtest EMA crossover strategy on TCS"
✅ "What is India VIX today and what does it mean?"
✅ "Scan for stocks with 90%+ confidence score"
✅ "Review my portfolio" (upload screenshot)
```

---

## 📁 Project Structure

```
AI_Trading/
├── agents/
│   ├── graph.py          # LangGraph pipeline
│   ├── nodes.py          # All agent node functions
│   ├── state.py          # Shared AgentState
│   └── llm_provider.py   # Groq LLM client
├── strategies/
│   ├── indicators.py     # pandas-ta indicators
│   └── setup_engine.py   # Trade setup analysis
├── backtester/
│   └── engine.py         # Walk-forward backtester
├── confidence/
│   └── engine.py         # Deterministic scoring engine
├── tools/
│   ├── equity_data.py    # yfinance live + historical
│   ├── mf_data.py        # mfapi.in mutual fund data
│   ├── fixed_income.py   # FD rates, G-Sec, RBI
│   └── news_data.py      # RSS feeds + sentiment
├── ui/
│   ├── streamlit_app.py  # Main Streamlit frontend
│   └── charts.py         # Plotly chart builders
├── db/
│   ├── models.py         # SQLAlchemy ORM models
│   └── database.py       # Async engine + sessions
├── core/
│   └── config.py         # Settings loader (.env + config.yaml)
├── tests/
│   ├── test_indicators.py
│   ├── test_confidence_gate.py
│   └── test_backtester.py
├── config.yaml           # All weights, thresholds, strategy params
├── requirements.txt
├── .env.example
├── main.py
└── pyproject.toml
```

---

## 🔒 API Keys Required

| Service | Free? | Purpose |
|---------|-------|---------|
| [Groq](https://console.groq.com) | ✅ Yes | LLM + Vision |
| yfinance | ✅ Yes | Market data (no key needed) |
| mfapi.in | ✅ Yes | Mutual fund NAVs |
| RSS feeds | ✅ Yes | Financial news |
| [Angel One SmartAPI](https://smartapi.angelbroking.com) | ✅ Yes | Live data + order routing |

---

## 🗺️ Next Steps

- [ ] Docker containerization (`Dockerfile` + `docker-compose.yml`)
- [ ] Telegram / WhatsApp alerts when watchlist hits 90%+
- [ ] More brokers: Upstox, Zerodha Kite
- [ ] Option chain analysis (NSE F&O)
- [ ] Portfolio optimization (Markowitz / Black-Litterman)
- [ ] Real-time WebSocket price feed
- [ ] Isotonic calibration of confidence scores
- [ ] Production PostgreSQL backend
- [ ] Scheduled market scan (cron / Celery)