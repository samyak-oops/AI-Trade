# Trading Strategies & Quantitative Models

This document details all technical indicators, trading strategies, price action setups, risk management rules, and quantitative models implemented in the **AI Investment & Trading Assistant**.

---

## 1. Technical Momentum & Trend Strategies

### 1.1 Triple EMA Trend Alignment
- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L83-L105) & [`strategies/indicators.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/indicators.py#L27-L29)
- **Parameters:** 20 EMA (fast), 50 EMA (medium), 200 EMA (long-term baseline).
- **Rules:**
  - **Golden Alignment (Bullish):** $\text{Price} > \text{EMA}_{20} > \text{EMA}_{50} > \text{EMA}_{200}$ $\rightarrow$ $+2$ BUY votes.
  - **Bearish Alignment (Short):** $\text{Price} < \text{EMA}_{20} < \text{EMA}_{50} < \text{EMA}_{200}$ $\rightarrow$ $+2$ SELL votes.
  - **Intermediate Pullback:** Price above 50 EMA ($+1$ BUY) or below 50 EMA ($+1$ SELL).

### 1.2 EMA Crossover Strategy
- **Implementation:** [`backtester/engine.py`](file:///c:/Users/samya/samyak/AI_Trading/backtester/engine.py#L103-L116)
- **Logic:**
  - Signal fires strictly on the crossover bar:
    - **Bullish:** $\text{EMA}_{20} > \text{EMA}_{50}$ and $\text{EMA}_{20}[t-1] \le \text{EMA}_{50}[t-1]$ ($\text{Signal} = 1$).
    - **Bearish:** $\text{EMA}_{20} < \text{EMA}_{50}$ and $\text{EMA}_{20}[t-1] \ge \text{EMA}_{50}[t-1]$ ($\text{Signal} = -1$).
  - Serves as the primary engine for walk-forward statistical backtesting.

### 1.3 Supertrend Volatility Trailing Stop
- **Implementation:** [`strategies/indicators.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/indicators.py#L82-L120)
- **Parameters:** ATR Period = 7, Multiplier = 3.0.
- **Rules:**
  - Dynamic upper/lower bands computed using median price $\frac{\text{High} + \text{Low}}{2} \pm 3 \times \text{ATR}$.
  - Bullish regime when price closes above upper band ($\text{Direction} = 1$, $+2$ BUY votes).
  - Bearish regime when price closes below lower band ($\text{Direction} = -1$, $+2$ SELL votes).

### 1.4 RSI Momentum & Mean Reversion
- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L107-L125)
- **Parameters:** 14-period RSI, Oversold = 30, Overbought = 70.
- **Rules:**
  - **Oversold Reversal:** $\text{RSI} < 30$ and turning upward ($\text{RSI}[t] > \text{RSI}[t-1]$) $\rightarrow$ $+2$ BUY votes.
  - **Overbought Reversal:** $\text{RSI} > 70$ and turning downward ($\text{RSI}[t] < \text{RSI}[t-1]$) $\rightarrow$ $+2$ SELL votes.

### 1.5 MACD Momentum Crossover
- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L126-L152)
- **Parameters:** Fast = 12, Slow = 26, Signal = 9.
- **Rules:**
  - **Bullish Crossover:** Histogram flips from negative to positive ($+2$ BUY votes).
  - **Bearish Crossover:** Histogram flips from positive to negative ($+2$ SELL votes).
  - **Momentum Bias:** $\text{MACD} > \text{Signal}$ and $\text{MACD} > 0$ ($+1$ BUY); $\text{MACD} < \text{Signal}$ and $\text{MACD} < 0$ ($+1$ SELL).

### 1.6 ADX Trend Strength Filter
- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L153-L164)
- **Parameters:** 14-period, Trend Threshold = 25.
- **Logic:**
  - Measures directional trend strength independently of direction.
  - When $\text{ADX} > 25$, confirms strong trend and unlocks full weighting for breakout/trend setups. Weak trend ($\text{ADX} \le 25$) is discounted by $50\%$.

### 1.7 52-Week High Breakout with Volume Surge
- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L179-L190)
- **Parameters:** Donchian High = 260 trading sessions (52 weeks $\times$ 5 bars), Volume Surge Multiplier = $1.5\times$ 20-day Volume MA.
- **Rules:**
  - Triggers $+3$ BUY votes when current price tests/exceeds 52-week high with $\text{Volume} \ge 1.5 \times \text{Volume}_{MA20}$.

### 1.8 Bollinger Bands Mean Reversion
- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L191-L205)
- **Parameters:** 20-period SMA, 2.0 Standard Deviations.
- **Rules:**
  - **Long Reversal:** $\text{Price} \le \text{BB}_{\text{lower}}$ and $\text{RSI} < 35$ ($+2$ BUY votes).
  - **Short Reversal:** $\text{Price} \ge \text{BB}_{\text{upper}}$ and $\text{RSI} > 65$ ($+2$ SELL votes).

---

## 2. Candlestick Patterns & Price Action

Automated pattern detection on 3-bar sequences ([`strategies/indicators.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/indicators.py#L202-L241)):

| Pattern | Type | Recognition Logic |
| :--- | :--- | :--- |
| **Hammer** | Bullish Reversal | Lower wick $> 2\times$ body; upper wick $< 0.3\times$ body. |
| **Inverted Hammer** | Bullish Reversal | Upper wick $> 2\times$ body; lower wick $< 0.3\times$ body. |
| **Bullish Engulfing** | Bullish Reversal | Prior red candle followed by green candle whose body fully covers prior candle. |
| **Bearish Engulfing** | Bearish Reversal | Prior green candle followed by red candle whose body fully covers prior candle. |
| **Shooting Star** | Bearish Reversal | Upper wick $> 2\times$ body; lower wick $< 0.3\times$ body; close $<$ open. |
| **Doji** | Indecision / Neutral | Candle body $< 10\%$ of total high-low candle range. |

---

## 3. Structural Levels (Pivots & Fibonacci)

### 3.1 Floor Pivot Points
- **Implementation:** [`strategies/indicators.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/indicators.py#L244-L260)
- Computed from previous session High ($H$), Low ($L$), and Close ($C$):
  - $\text{Pivot } (P) = \frac{H + L + C}{3}$
  - $R_1 = 2P - L \quad\vert\quad S_1 = 2P - H$
  - $R_2 = P + (H - L) \quad\vert\quad S_2 = P - (H - L)$
  - $R_3 = H + 2(P - L) \quad\vert\quad S_3 = L - 2(H - P)$

### 3.2 Fibonacci Retracement Levels
- Projected swing levels: $23.6\%$, $38.2\%$, $50.0\%$, $61.8\%$, $78.6\%$, $100.0\%$.

### 3.3 Visual Pattern Recognition (Vision LLM)
- Multimodal Vision Agent analyzes user-uploaded chart screenshots for chart patterns (Head & Shoulders, Double Top/Bottom, Triangles, Channels).

---

## 4. Multi-Timeframe (MTF) Trend Alignment

- **Implementation:** [`strategies/setup_engine.py`](file:///c:/Users/samya/samyak/AI_Trading/strategies/setup_engine.py#L258-L266)
- Evaluates Daily trade setups against the Weekly 50 EMA:
  - If Daily setup is **BUY** but Weekly Close is below Weekly 50 EMA: flagged as counter-trend penalty.
  - If Daily setup is **SELL** but Weekly Close is above Weekly 50 EMA: flagged as counter-trend penalty.

---

## 5. Risk Management & Position Sizing

- **Implementation:** [`config.yaml`](file:///c:/Users/samya/samyak/AI_Trading/config.yaml#L36-L42) & [`agents/nodes.py`](file:///c:/Users/samya/samyak/AI_Trading/agents/nodes.py#L274-L291)

### 5.1 Dynamic ATR Stop-Loss & Targets
* $\text{ATR}$ computed using 14-period True Range.
* **For BUY (Long):**
  $$\text{Stop Loss} = \text{Entry} - (2.0 \times \text{ATR})$$
  $$\text{Target 1} = \text{Entry} + (4.0 \times \text{ATR}) \quad (\text{Risk-to-Reward } 1:2)$$
  $$\text{Target 2} = \text{Entry} + (6.0 \times \text{ATR}) \quad (\text{Risk-to-Reward } 1:3)$$
* **For SELL (Short):**
  $$\text{Stop Loss} = \text{Entry} + (2.0 \times \text{ATR})$$
  $$\text{Target 1} = \text{Entry} - (4.0 \times \text{ATR}) \quad (\text{Risk-to-Reward } 1:2)$$
  $$\text{Target 2} = \text{Entry} - (6.0 \times \text{ATR}) \quad (\text{Risk-to-Reward } 1:3)$$

### 5.2 1% Fixed Capital Risk Position Sizing
$$\text{Shares to Buy/Sell} = \left\lfloor \frac{\text{Portfolio Capital} \times 0.01}{|\text{Entry Price} - \text{Stop Loss}|} \right\rfloor$$

### 5.3 Trading Session Filter
* **Opening Bell Filter:** Avoid entries in the first 15 minutes (9:15–9:30 AM IST).
* **Closing Bell Filter:** Avoid entries in the last 15 minutes (3:15–3:30 PM IST).

---

## 6. Quantitative Walk-Forward Backtester

- **Implementation:** [`backtester/engine.py`](file:///c:/Users/samya/samyak/AI_Trading/backtester/engine.py)
- Powered by **`vectorbt`** (with automated manual fallback).
- **Data Period:** 5-year daily OHLCV.
- **Walk-Forward Split:** 70% In-Sample training, 30% Out-of-Sample validation.
- **Frictional Costs:**
  - Brokerage: $0.03\%$ per leg.
  - STT: $0.1\%$ on sell leg.
  - Slippage: $0.05\%$ per execution.
- **Pass Requirements:**
  - Total in-sample trades $\ge 30$.
  - Out-of-sample trades $\ge 5$ with positive expectancy ($\text{Expectancy} > 0$).

---

## 7. Deterministic 7-Pillar Confidence Gate

- **Implementation:** [`confidence/engine.py`](file:///c:/Users/samya/samyak/AI_Trading/confidence/engine.py)
- The LLM **never** decides the trade score. A deterministic engine computes a weighted score between $0\%$ and $100\%$:

$$\text{Confidence Score} = \sum_{i=1}^{7} (W_i \times S_i)$$

| Pillar | Weight | Factors Evaluated |
| :--- | :---: | :--- |
| **Indicator Confluence** | **25%** | Agreement between EMA, RSI, MACD, Supertrend, BB, and Donchian. |
| **Backtest Edge** | **25%** | Win rate ($\ge 55\%$), Profit Factor ($\ge 2.0$), Sharpe ratio, and expectancy. |
| **Multi-Timeframe** | **15%** | Alignment between Daily execution and Weekly higher-timeframe trend. |
| **Market Regime** | **10%** | Nifty 50 trend, India VIX ($<15$ calm, $15\text{--}20$ normal, $>24$ volatile), and FII/DII flow. |
| **Fundamentals & News** | **10%** | P/E, Debt/Equity, ROE, FinBERT sentiment score, and earnings event risk. |
| **Agent Track Record** | **10%** | Historical hit-rate of previous signals logged in SQLite database. |
| **Liquidity Quality** | **5%** | Average traded volume, circuit-limit checks, and spread quality. |

### Hard Fail Rules (Automatic NO-TRADE):
1. Score $\le 75.0\%$ (or $\le 90.0\%$ in strict mode).
2. Risk-to-Reward ratio $< 1:2$.
3. Backtest $< 30$ trades or negative out-of-sample expectancy.
4. Price data $> 5$ minutes stale during market hours.
5. High-impact event risk (earnings release, RBI policy) within the trade window.
