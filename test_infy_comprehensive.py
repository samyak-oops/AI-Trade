import asyncio
import sys
import io

# Force UTF-8 for console output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from langchain_core.messages import HumanMessage
from agents.graph import get_graph
from tools.equity_data import get_historical_data, get_live_price
from strategies.indicators import add_all_indicators
from core.config import get_settings

async def main():
    print("==================================================================")
    print("1. TESTING STOCK DATA & TECHNICAL INDICATORS FOR INFY")
    print("==================================================================")
    cfg = get_settings().yaml_section("strategies")
    df = get_historical_data("INFY", "NSE", period="1y", interval="1d")
    print(f"Historical data shape: {df.shape}")
    if df.empty:
        print("[ERROR] No historical data retrieved for INFY!")
        return
        
    df_ind = add_all_indicators(df, cfg)
    print(f"Computed columns: {list(df_ind.columns)}")
    last = df_ind.iloc[-1]
    print(f"Latest Date: {df_ind.index[-1]}")
    print(f"Close: {last['Close']:.2f}")
    print(f"EMA 20: {last.get('EMA_20', 0):.2f} | EMA 50: {last.get('EMA_50', 0):.2f} | EMA 200: {last.get('EMA_200', 0):.2f}")
    print(f"RSI: {last.get('RSI', 0):.2f}")
    print(f"MACD: {last.get('MACD', 0):.2f} | MACD Signal: {last.get('MACD_signal', 0):.2f} | MACD Hist: {last.get('MACD_hist', 0):.2f}")
    print(f"ATR: {last.get('ATR', 0):.2f}")
    print(f"Supertrend Direction: {last.get('Supertrend_dir', 0)}")

    print("\n==================================================================")
    print("2. RUNNING MULTI-AGENT ANALYSIS PIPELINE FOR INFY")
    print("==================================================================")
    graph = get_graph()
    initial_state = {
        "messages": [HumanMessage(content="Perform full swing trading analysis, technical setup, backtest validation, confidence assessment, and complete signal report for INFY (Infosys).")],
        "user_query": "Analyze INFY for swing trading with complete signal report, chart analysis and backtesting.",
        "symbol": "INFY",
        "session_id": "test_infy_session",
        "capital": 100_000.0,
    }
    config = {"configurable": {"thread_id": "test_infy_session"}}
    
    result = await graph.ainvoke(initial_state, config=config)
    print("\n>>> Pipeline execution completed successfully.")
    
    # 1. Setup details
    setup = result.get("trade_setup")
    print("\n--- [A] TRADE SETUP ---")
    if setup:
        print(f"Direction:    {setup.direction}")
        print(f"Entry Price:  Rs. {setup.entry_price:.2f}")
        print(f"Stop Loss:    Rs. {setup.stop_loss:.2f}")
        print(f"Target 1:     Rs. {setup.target1:.2f}")
        print(f"Target 2:     Rs. {setup.target2:.2f}")
        print(f"Risk:Reward:  1:{setup.risk_reward:.2f}")
        print(f"Timeframe:    {setup.timeframe}")
        print(f"ATR:          Rs. {setup.atr:.2f}")
        print(f"Strategies:   {setup.strategies_triggered}")
        print(f"Candlesticks: {setup.candlestick_patterns}")
        print(f"Warnings:     {setup.warnings}")
    else:
        print("No trade setup returned.")

    # 2. Backtest details
    bt = result.get("backtest_result")
    print("\n--- [B] BACKTEST VALIDATION ---")
    if bt:
        print(f"Total Trades:     {bt.total_trades}")
        print(f"Win Rate:         {bt.win_rate:.1f}%")
        print(f"CAGR:             {bt.cagr:.1f}%")
        print(f"Sharpe Ratio:     {bt.sharpe:.2f}")
        print(f"Max Drawdown:     {bt.max_drawdown:.1f}%")
        print(f"Profit Factor:    {bt.profit_factor:.2f}")
        print(f"Expectancy:       {bt.expectancy:.2f}%")
        print(f"Buy & Hold:       {bt.buy_hold_return_pct:.1f}%")
        print(f"Is Valid (>=30):  {bt.is_valid}")
        print(f"OOS Valid:        {bt.is_out_of_sample_valid}")
        print(f"Warnings:         {bt.warnings}")
    else:
        print("No backtest result returned.")

    # 3. Confidence engine
    conf = result.get("confidence_report")
    print("\n--- [C] CONFIDENCE ENGINE REPORT ---")
    if conf:
        print(f"Confidence Score: {conf.score:.1f}%")
        print(f"Passed Gate:      {conf.passed_gate()}")
        print("Component Scores:")
        for k, v in conf.breakdown.items():
            print(f"  - {k}: {v:.1f}%")
        if conf.hard_rule_failures:
            print(f"Hard Rule Failures: {conf.hard_rule_failures}")
        print(f"Reasons High:     {conf.reasons_high}")
        print(f"Reasons Low:      {conf.reasons_low}")
    else:
        print("No confidence score returned.")

    # 4. Position Sizing
    pos = result.get("position_size", 0.0)
    print(f"\n--- [D] RISK & POSITION SIZING ---")
    print(f"Calculated Position Size: {pos:.0f} shares")

    # 5. Synthesizer final output
    print("\n--- [E] FINAL SYNTHESIZED SIGNAL REPORT ---")
    final_output = result.get("final_response", "")
    print(final_output)

if __name__ == "__main__":
    asyncio.run(main())
