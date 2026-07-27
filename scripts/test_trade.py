"""One-shot test trade — places a USDJPY buy, waits 10s, closes it. Verifies MT5 pipeline."""
import sys, time
sys.path.insert(0, r"C:\Users\anton\Documents\trading-bot")

import MetaTrader5 as mt5

mt5.initialize()
info = mt5.account_info()
if not info:
    print("MT5 not connected"); sys.exit(1)

print(f"Account: {info.login} | Balance: {info.balance} | Server: {info.server}")

sym = "USDJPY"
mt5.symbol_select(sym, True)
tick = mt5.symbol_info_tick(sym)
price = tick.ask
sl    = round(price - 1.0, 3)   # 100-pip stop
tp    = round(price + 1.0, 3)   # 100-pip target

req = {
    "action":    mt5.TRADE_ACTION_DEAL,
    "symbol":    sym,
    "volume":    0.01,
    "type":      mt5.ORDER_TYPE_BUY,
    "price":     price,
    "sl":        sl,
    "tp":        tp,
    "deviation": 20,
    "magic":     999999,
    "comment":   "AiDEN_TEST",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(req)
print(f"Order result: retcode={result.retcode} | ticket={result.order} | comment={result.comment}")

if result.retcode == mt5.TRADE_RETCODE_DONE:
    print(f"OPEN: ticket {result.order} at {price}")
    time.sleep(10)
    pos = mt5.positions_get(ticket=result.order)
    if pos:
        close_req = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    sym,
            "volume":    0.01,
            "type":      mt5.ORDER_TYPE_SELL,
            "position":  result.order,
            "price":     mt5.symbol_info_tick(sym).bid,
            "deviation": 20,
            "magic":     999999,
            "comment":   "AiDEN_TEST_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        cr = mt5.order_send(close_req)
        print(f"CLOSE: retcode={cr.retcode} | comment={cr.comment}")
    else:
        print("Position already gone (TP/SL hit or already closed)")
else:
    print(f"FAILED: {result.comment}")

mt5.shutdown()
