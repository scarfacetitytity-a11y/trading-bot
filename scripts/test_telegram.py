"""Quick Telegram smoke test — run from trading-bot root.

Usage:  python test_telegram.py
Sends a fake trade open alert and a trade close alert to confirm the bot
can reach Telegram. Does NOT touch MT5 or place any orders.
"""
import pandas as pd
import numpy as np
from execution import telegram_notify as tg

# Fake M15 bars for chart generation
np.random.seed(42)
closes = 1.28 + np.cumsum(np.random.randn(60) * 0.0003)
df = pd.DataFrame({
    "open":  closes - 0.0001,
    "high":  closes + 0.0005,
    "low":   closes - 0.0005,
    "close": closes,
    "tick_volume": np.ones(60) * 100,
})

print("Sending TRADE OPEN alert...")
tg.notify_trade_open(
    symbol    = "GBPUSD",
    direction = -1,
    score     = 7,
    entry     = 1.2840,
    sl        = 1.2870,
    tp        = 1.2780,
    lots      = 0.10,
    equity    = 10000.0,
    atr       = 0.0012,
    df        = df,
    reasons   = ["Equal lows 1.2780 (3-touch)", "Order block M5", "H4 SHORT bias", "London sweep"],
)
print("Trade open sent.")

import time
time.sleep(2)

print("Sending TRADE CLOSE alert...")
tg.notify_trade_close(
    symbol      = "GBPUSD",
    direction   = -1,
    outcome     = "win",
    r_multiple  = 2.0,
    pnl_usd     = 200.0,
    equity      = 10200.0,
    session_pnl = 200.0,
)
print("Trade close sent.")
print("\nDone. Check your Telegram.")
