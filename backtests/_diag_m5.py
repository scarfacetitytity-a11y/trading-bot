"""Test max bar count for copy_rates_from_pos on FTMO demo."""
import time
import pandas as pd
import MetaTrader5 as mt5

mt5.initialize()
time.sleep(1)

mt5.symbol_select("XAUUSD", True)
time.sleep(0.3)

for n in [1000, 10_000, 50_000, 99_000, 100_000, 150_000, 200_000]:
    r = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, n)
    if r is None:
        print(f"  {n:>8,} bars: FAILED  err={mt5.last_error()}")
    else:
        df = pd.DataFrame(r)
        first = pd.to_datetime(df["time"].iloc[0], unit="s").date()
        last  = pd.to_datetime(df["time"].iloc[-1], unit="s").date()
        print(f"  {n:>8,} bars: OK  {first} -> {last}")

mt5.shutdown()
