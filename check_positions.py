import MetaTrader5 as mt5
mt5.initialize()
positions = mt5.positions_get()
if positions:
    for p in positions:
        d = "LONG" if p.type == 0 else "SHORT"
        print(p.symbol, d, p.volume, "lots | entry", round(p.price_open, 5), "| profit", round(p.profit, 2), "| SL", round(p.sl, 5), "| TP", round(p.tp, 5))
else:
    print("No open positions")
mt5.shutdown()
