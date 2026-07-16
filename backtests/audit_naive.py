"""NAIVE execution audit — bisect signal edge vs. management inflation.

Runs the EXACT strategy signals (generate_signals) through dead-simple execution:
  - Entry: edge-trigger (signal 0 -> non-zero), fill at NEXT bar open
  - Stop:  fixed initial SL from strat._stops at the signal bar — NEVER moves
  - TP:    fixed at entry +/- rr_target * risk_distance
  - Intrabar: SL checked BEFORE TP (worst case — no optimistic fills)
  - NOTHING ELSE: no T1 partial, no trailing, no TradeManager, no TP extension

R is PURE: R = (exit - entry) / |entry - initial_sl| for long (sign-flipped short).
No size multiplier. A trade stopped at its initial SL = exactly -1.00R.

Two honest rates reported separately:
  - hit_tp_rate    : fraction of trades that reached the TP (real target win)
  - profitable_rate: fraction with R > 0 (any positive close)

If naive STILL prints huge expectancy -> lookahead lives in generate_signals.
If naive normalises (WR ~45-55%, small expectancy) -> inflation is management.

Usage:
    python -m backtests.audit_naive
    python -m backtests.audit_naive XAUUSD --trace   # trace highest-R trade
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.tm_backtest import _ALL_TM_SYMBOLS
from backtests.run_stress_test import _build_strategy, _load_symbol

LOGS = Path(__file__).resolve().parent.parent / "logs"


@dataclass
class Trade:
    symbol:     str
    direction:  int
    entry_time: pd.Timestamp
    exit_time:  pd.Timestamp
    entry:      float
    initial_sl: float
    tp:         float
    exit:       float
    R:          float
    reason:     str      # "SL" | "TP" | "EOD" (end of data, still open)
    bars:       int


def naive_run(symbol: str) -> list[Trade]:
    df15, _ = _load_symbol(symbol)          # M15 only — pure signal test
    if df15 is None:
        return []

    strat   = _build_strategy(symbol)
    signals = strat.generate_signals(df15)  # the EXACT algo
    stops   = getattr(strat, "_stops", pd.Series(np.nan, index=df15.index))
    rr      = float(getattr(strat, "rr_target", 2.5))

    times = pd.to_datetime(df15["time"], utc=True).reset_index(drop=True)
    op    = df15["open"].astype(float).values
    hi    = df15["high"].astype(float).values
    lo    = df15["low"].astype(float).values

    sig   = signals.values
    slv   = stops.values

    trades: list[Trade] = []

    pos        = 0
    entry_px   = 0.0
    init_sl    = 0.0
    tp_px      = 0.0
    entry_i    = 0

    for i in range(1, len(df15)):
        # ── Manage open position (SL before TP — worst case) ──
        if pos != 0:
            sl_hit = (pos == 1 and lo[i] <= init_sl) or (pos == -1 and hi[i] >= init_sl)
            tp_hit = (pos == 1 and hi[i] >= tp_px)   or (pos == -1 and lo[i] <= tp_px)

            exit_px = None
            reason  = None
            if sl_hit:
                exit_px, reason = init_sl, "SL"
            elif tp_hit:
                exit_px, reason = tp_px, "TP"

            if exit_px is not None:
                rd   = abs(entry_px - init_sl)
                move = (exit_px - entry_px) if pos == 1 else (entry_px - exit_px)
                R    = move / rd if rd > 1e-12 else 0.0
                trades.append(Trade(
                    symbol=symbol, direction=pos,
                    entry_time=times.iloc[entry_i], exit_time=times.iloc[i],
                    entry=entry_px, initial_sl=init_sl, tp=tp_px,
                    exit=exit_px, R=R, reason=reason, bars=i - entry_i,
                ))
                pos = 0

        # ── New entry — edge-trigger, fill at THIS bar's open ──
        if pos == 0:
            s_prev = sig[i - 1]
            s_prev2 = sig[i - 2] if i >= 2 else 0.0
            sl_lvl = slv[i - 1]
            if s_prev != 0 and s_prev2 == 0 and not np.isnan(sl_lvl) and sl_lvl > 0:
                direction = 1 if s_prev > 0 else -1
                e_px      = op[i]
                rd        = abs(e_px - sl_lvl)
                if rd > 1e-9:
                    pos      = direction
                    entry_px = e_px
                    init_sl  = sl_lvl
                    tp_px    = (e_px + rr * rd) if direction == 1 else (e_px - rr * rd)
                    entry_i  = i

    return trades


def summarise(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    R      = np.array([t.R for t in trades])
    wins   = R[R > 0]
    losses = R[R <= 0]
    n_tp   = sum(1 for t in trades if t.reason == "TP")
    n_sl   = sum(1 for t in trades if t.reason == "SL")

    return {
        "n_trades":        len(trades),
        "n_TP":            n_tp,
        "n_SL":            n_sl,
        "hit_tp_rate":     n_tp / len(trades),
        "profitable_rate": float((R > 0).mean()),
        "avg_R":           float(R.mean()),
        "avg_win_R":       float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R":      float(losses.mean()) if len(losses) else 0.0,
        "expectancy_R":    float(R.mean()),
        "sum_R":           float(R.sum()),
        "worst_R":         float(R.min()),
        "best_R":          float(R.max()),
    }


def trace_highest(trades: list[Trade], symbol: str) -> None:
    if not trades:
        print("  no trades to trace")
        return
    t = max(trades, key=lambda x: x.R)
    print(f"\n  === HIGHEST-R TRADE TRACE  {symbol} ===")
    print(f"  dir={t.direction:+d}  entry={t.entry:.5f} @ {t.entry_time}")
    print(f"  initial_SL={t.initial_sl:.5f}  TP={t.tp:.5f}")
    print(f"  exit={t.exit:.5f} ({t.reason}) @ {t.exit_time}  after {t.bars} bars")
    rd = abs(t.entry - t.initial_sl)
    print(f"  risk_distance={rd:.5f}  ->  R={t.R:.3f}")
    # sanity: TP distance must equal rr * rd
    tp_dist = abs(t.tp - t.entry)
    print(f"  TP_distance/risk_distance = {tp_dist/rd:.3f}  (should equal rr_target)")
    if t.reason == "TP":
        print(f"  Expected R at TP = +{tp_dist/rd:.3f}  |  logged R = {t.R:.3f}  "
              f"{'OK' if abs(t.R - tp_dist/rd) < 0.05 else 'MISMATCH!'}")


def main(symbols=None, trace=False, save_csv=True) -> None:
    if symbols is None:
        symbols = _ALL_TM_SYMBOLS

    print(f"\n{'='*88}")
    print(f"  NAIVE EXECUTION AUDIT — exact signals, fixed SL/TP, no management")
    print(f"  If expectancy stays huge -> lookahead in signal.  If it normalises -> management.")
    print(f"{'='*88}")
    print(f"  {'Symbol':<14} {'Trades':>7} {'hitTP%':>7} {'prof%':>7} "
          f"{'avgR':>7} {'winR':>7} {'lossR':>7} {'expR':>7} {'sumR':>9} {'worstR':>7}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")

    all_trades: list[Trade] = []
    for symbol in symbols:
        trades = naive_run(symbol)
        all_trades.extend(trades)
        s = summarise(trades)
        if not s:
            print(f"  {symbol:<14}  no trades")
            continue
        print(f"  {symbol:<14} {s['n_trades']:>7} {100*s['hit_tp_rate']:>6.1f}% "
              f"{100*s['profitable_rate']:>6.1f}% {s['avg_R']:>+7.3f} "
              f"{s['avg_win_R']:>+7.2f} {s['avg_loss_R']:>+7.2f} {s['expectancy_R']:>+7.3f} "
              f"{s['sum_R']:>+9.1f} {s['worst_R']:>+7.2f}")
        if trace:
            trace_highest(trades, symbol)

    # ── Portfolio aggregate ──
    agg = summarise(all_trades)
    if agg:
        print(f"  {'-'*14}")
        print(f"  {'PORTFOLIO':<14} {agg['n_trades']:>7} {100*agg['hit_tp_rate']:>6.1f}% "
              f"{100*agg['profitable_rate']:>6.1f}% {agg['avg_R']:>+7.3f} "
              f"{agg['avg_win_R']:>+7.2f} {agg['avg_loss_R']:>+7.2f} {agg['expectancy_R']:>+7.3f} "
              f"{agg['sum_R']:>+9.1f} {agg['worst_R']:>+7.2f}")

    # ── Integrity checks ──
    print(f"\n  === INTEGRITY CHECKS ===")
    sl_trades = [t for t in all_trades if t.reason == "SL"]
    bad_sl    = [t for t in sl_trades if t.R > -0.98]   # SL should be ~ -1.00R
    print(f"  SL trades logged as ~-1R : {len(sl_trades) - len(bad_sl)}/{len(sl_trades)} correct")
    if bad_sl:
        print(f"  !! {len(bad_sl)} SL trades with R > -0.98 (possible mislabel):")
        for t in bad_sl[:5]:
            print(f"     {t.symbol} R={t.R:.3f} entry={t.entry:.4f} sl={t.initial_sl:.4f} exit={t.exit:.4f}")
    tp_trades = [t for t in all_trades if t.reason == "TP"]
    bad_tp    = [t for t in tp_trades if t.R < 0]        # TP must be positive R
    print(f"  TP trades with positive R: {len(tp_trades) - len(bad_tp)}/{len(tp_trades)} correct")

    if save_csv and all_trades:
        LOGS.mkdir(exist_ok=True)
        rows = [{
            "symbol": t.symbol, "direction": t.direction,
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "entry": t.entry, "initial_sl": t.initial_sl, "tp": t.tp,
            "exit": t.exit, "R": round(t.R, 4), "reason": t.reason, "bars": t.bars,
        } for t in all_trades]
        out = LOGS / "audit_naive_trades.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  Per-trade log saved: {out}  ({len(rows)} trades)")

    print(f"{'='*88}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--trace", action="store_true", help="Trace highest-R trade per symbol")
    args = ap.parse_args()
    main(symbols=args.symbols or None, trace=args.trace)
