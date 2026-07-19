"""Quick sweep of RSI thresholds on the Sniper to find the best win rate."""
import pandas as pd
from backtests.engine import Backtest
from strategies.sniper import SniperStrategy

df = pd.read_csv("data/processed/XAUUSD_H1.csv")

print(f"\n{'RSI Threshold':<16} {'Win Rate':>9} {'Trades':>8} {'Return':>9} {'PF':>7} {'Sharpe':>8}")
print("-" * 60)

for rsi in [40, 35, 30, 25, 20]:
    strat = SniperStrategy(rsi_oversold=rsi, rsi_overbought=100 - rsi)
    bt    = Backtest(strat, initial_capital=10_000)
    r     = bt.run(df, symbol="XAUUSD")
    m     = r.metrics
    print(
        f"RSI {rsi}/{100-rsi:<8}    "
        f"{m['win_rate_pct']:>8.1f}%"
        f"{m['total_trades']:>9}"
        f"{m['total_return_pct']:>+9.1f}%"
        f"{m['profit_factor']:>8.3f}"
        f"{m['sharpe_ratio']:>9.3f}"
    )
