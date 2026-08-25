"""Full Market Reader diagnostic — shows exactly why signals pass or block."""
import sys, math
sys.path.insert(0, ".")
import MetaTrader5 as mt5
import pandas as pd
from strategies.aiden_index import AiDENIndexStrategy
from core.instrument_profile import build_strategy_kwargs
from execution.market_reader import read_market
from core.amd_detector import AMDDetector

mt5.initialize()

SYMBOLS = ["XAUUSD", "US100.cash", "US30.cash", "EURUSD"]

for sym in SYMBOLS:
    bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 300)
    if bars is None or len(bars) < 50:
        print(f"{sym}: NO DATA")
        continue

    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.reset_index(drop=True)

    kw = build_strategy_kwargs(sym)
    strat = AiDENIndexStrategy(**kw)
    signals = strat.generate_signals(df)

    last_signal = float(signals.iloc[-1])
    desired = int(last_signal)

    try:
        h4_bias = strat.current_h4_bias(df)
    except Exception as e:
        h4_bias = 0

    # ATR
    try:
        _atr = float(getattr(strat, "_atr_cache", pd.Series()).iloc[-1])
        if math.isnan(_atr): _atr = 0.0
    except Exception:
        _atr = 0.0

    # AMD sweep
    amd = AMDDetector()
    try:
        amd.update(df)
        _sweep = amd.last_sweep()
    except Exception:
        _sweep = None

    # Weekly mid
    _w1_mid = float("nan")
    try:
        df_idx = df.copy()
        df_idx.index = df_idx["time"]
        _wk = df_idx.resample("W-MON").agg({"high": "max", "low": "min"})
        if len(_wk) >= 1:
            _wh = float(_wk["high"].iloc[-1])
            _wl = float(_wk["low"].iloc[-1])
            if not (math.isnan(_wh) or math.isnan(_wl)):
                _w1_mid = (_wh + _wl) / 2.0
    except Exception:
        pass

    tick = mt5.symbol_info_tick(sym)
    price = tick.bid if tick else 0.0

    print(f"\n{'='*60}")
    print(f"{sym} | strategy signal: {last_signal:+.0f} | H4 bias: {h4_bias} | ATR: {_atr:.4f} | price: {price}")
    print(f"  AMD sweep: {_sweep}")
    print(f"  w1_mid: {_w1_mid:.4f}" if not math.isnan(_w1_mid) else "  w1_mid: NaN")

    # Run Market Reader for each meaningful desired direction
    dirs = [d for d in ([desired] if desired != 0 else [1, -1])]
    if desired != 0:
        dirs = [desired]
    else:
        # show both to understand which side is closer
        dirs = [1, -1]

    for d in dirs:
        narrative = read_market(
            symbol=sym, desired=d,
            df_m15=df, df_m5=None, h4_bias=h4_bias,
            atr=_atr, amd_sweep=_sweep,
            w1_mid=_w1_mid,
        )
        label = "LONG" if d == 1 else "SHORT"
        verdict = "TRADE" if narrative.trade_bias != 0 else "BLOCK"
        print(f"  MR {label}: {verdict} | vote={narrative.vote} | phase={narrative.phase}")
        if narrative.reasons:
            print(f"    + {narrative.reasons}")
        if narrative.blocking:
            print(f"    X {narrative.blocking}")

mt5.shutdown()
