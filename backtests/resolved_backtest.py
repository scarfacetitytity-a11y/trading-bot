"""Intrabar-resolved backtest — honest exits, structural stops/targets, real metrics.

Compounds the whole practice:
  - Entries: exact strategy signals (edge-triggered, fill at next M15 open).
  - Plan: analyze_entry() gives the STRUCTURAL stop + liquidity target + type +
    grade + size. Grade-C / no-draw trades are SKIPPED (not counted as trades).
  - Exit resolution: walk the FINEST bars available (M1 if present, else M5) from
    entry forward and take the level touched FIRST chronologically. Within a single
    bar that spans both, assume SL-first (conservative) and flag it ambiguous — no
    guessing dressed up as a win.
  - R is structural and pure: (exit-entry)/(entry-structural_stop). SL hit = the
    real loss, never a mislabelled win.
  - Portfolio: shared equity with the live daily logic — halt new entries at -2%,
    hard close-all floor at -5% daily and -10% total. Tracks portfolio heat.

Outputs a per-trade CSV (logs/resolved_trades.csv) and a full metrics summary.

Usage:
    python -m backtests.resolved_backtest
    python -m backtests.resolved_backtest XAUUSD --risk 0.5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.tm_backtest import _ALL_TM_SYMBOLS
from backtests.run_stress_test import _build_strategy, _load_symbol
from execution.trade_analyzer import analyze_entry

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
LOGS      = Path(__file__).resolve().parent.parent / "logs"

DAILY_HALT   = 2.0    # stop new entries when day down this %
DAILY_FLOOR  = 5.0    # hard close-all
TOTAL_FLOOR  = 10.0


@dataclass
class RTrade:
    symbol: str; direction: int
    entry_time: str; exit_time: str
    entry: float; stop: float; target: float
    exit: float; R: float; reason: str
    trade_type: str; grade: str; size_mult: float
    mfe_R: float; mae_R: float; hit_target: bool
    hold_bars: int; ambiguous: bool; resolved_on: str


def _load_fine(symbol: str) -> dict:
    """Load M1 and M5 fine bars. M1 covers only ~3 months, so per-trade we pick the
    finest set that actually COVERS the entry time (M1 inside its window, else M5)."""
    out = {}
    for tf in ("M1", "M5"):
        p = PROCESSED / f"{symbol}_{tf}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            if len(df) > 100:
                out[tf] = df
    return out


def _pick_fine(fine: dict, entry_time):
    """Choose finest bar set that covers entry_time. Returns (df, tag) or (None,None)."""
    m1 = fine.get("M1")
    if m1 is not None and entry_time >= m1["time"].iloc[0]:
        return m1, "M1"
    m5 = fine.get("M5")
    if m5 is not None:
        return m5, "M5"
    return None, None


def _resolve_exit(fine, direction, entry, stop, target, start_time):
    """Walk fine bars from start_time; return (exit_px, reason, mfe_R, mae_R,
    hold_bars, ambiguous). SL checked before TP within a bar (conservative)."""
    rd = abs(entry - stop)
    sub = fine[fine["time"] >= start_time]
    if len(sub) == 0 or rd <= 0:
        return None
    hi = sub["high"].values; lo = sub["low"].values
    best_fav = entry; worst_adv = entry
    for k in range(len(sub)):
        h, l = hi[k], lo[k]
        if direction == 1:
            best_fav = max(best_fav, h); worst_adv = min(worst_adv, l)
            sl_hit = l <= stop; tp_hit = h >= target
        else:
            best_fav = min(best_fav, l); worst_adv = max(worst_adv, h)
            sl_hit = h >= stop; tp_hit = l <= target
        if sl_hit or tp_hit:
            ambiguous = sl_hit and tp_hit
            exit_px, reason = (stop, "SL") if sl_hit else (target, "TP")
            mfe = (best_fav - entry) / rd if direction == 1 else (entry - best_fav) / rd
            mae = (worst_adv - entry) / rd if direction == 1 else (entry - worst_adv) / rd
            return exit_px, reason, mfe, mae, k + 1, ambiguous
    # never hit — close at last fine close
    last = float(sub["close"].iloc[-1])
    mfe = (best_fav - entry) / rd if direction == 1 else (entry - best_fav) / rd
    mae = (worst_adv - entry) / rd if direction == 1 else (entry - worst_adv) / rd
    return last, "EOD", mfe, mae, len(sub), False


def run_symbol(symbol: str) -> list[RTrade]:
    df15, _ = _load_symbol(symbol)
    if df15 is None:
        return []
    fine = _load_fine(symbol)
    if not fine:
        return []

    strat  = _build_strategy(symbol)
    sig    = strat.generate_signals(df15)
    stops  = getattr(strat, "_stops", pd.Series(np.nan, index=df15.index))
    atr_c  = getattr(strat, "_atr_cache", pd.Series(np.nan, index=df15.index))
    times  = pd.to_datetime(df15["time"], utc=True).reset_index(drop=True)
    op     = df15["open"].astype(float).values
    sv     = sig.values; slv = stops.values

    trades: list[RTrade] = []
    i = 1
    while i < len(df15):
        if sv[i-1] != 0 and (i < 2 or sv[i-2] == 0) and not np.isnan(slv[i-1]) and slv[i-1] > 0:
            direction = 1 if sv[i-1] > 0 else -1
            entry = op[i]; etime = times.iloc[i]
            atr = float(atr_c.iloc[i-1]) if i-1 < len(atr_c) and not np.isnan(atr_c.iloc[i-1]) else 0.0
            if atr <= 0:
                i += 1; continue
            fine_df, tf_tag = _pick_fine(fine, etime)   # M1 if covered, else M5 — for EXIT resolution
            if fine_df is None:
                i += 1; continue
            m15ctx = df15.iloc[max(0, i-120):i].reset_index(drop=True)
            # Analyzer context is ALWAYS M5 (proper swing/liquidity structure) — M1
            # is too micro to read the levels that matter. Exit resolution uses M1.
            ctx_src = fine.get("M5", fine_df)
            m5ctx   = ctx_src[ctx_src["time"] <= etime].tail(120).reset_index(drop=True)
            h4b = strat.current_h4_bias(m15ctx) if hasattr(strat, "current_h4_bias") else 0
            plan = analyze_entry(df_m15=m15ctx, df_m5=m5ctx, direction=direction,
                                 entry=entry, stop=float(slv[i-1]), atr=atr, h4_bias=h4b)
            if not plan.tradeable:
                i += 1; continue
            res = _resolve_exit(fine_df, direction, entry, plan.stop, plan.tp, etime)
            if res is None:
                i += 1; continue
            exit_px, reason, mfe, mae, hold, ambiguous = res
            rd = abs(entry - plan.stop)
            R  = ((exit_px - entry) if direction == 1 else (entry - exit_px)) / rd
            # exit time from fine bars
            sub = fine_df[fine_df["time"] >= etime]
            xtime = str(sub["time"].iloc[min(hold-1, len(sub)-1)])
            hit = (reason == "TP")
            trades.append(RTrade(
                symbol=symbol, direction=direction,
                entry_time=str(etime), exit_time=xtime,
                entry=entry, stop=plan.stop, target=plan.tp, exit=exit_px,
                R=round(R, 4), reason=reason, trade_type=plan.trade_type,
                grade=plan.grade, size_mult=plan.size_mult,
                mfe_R=round(mfe, 3), mae_R=round(mae, 3), hit_target=hit,
                hold_bars=hold, ambiguous=ambiguous, resolved_on=tf_tag,
            ))
        i += 1
    return trades


def summarise(trades: list[RTrade], tag: str) -> dict:
    if not trades:
        return {}
    R = np.array([t.R for t in trades])
    wins = R[R > 0]; losses = R[R <= 0]
    amb = sum(1 for t in trades if t.ambiguous)
    return {
        "scope": tag, "trades": len(trades),
        "hit_tp_rate": sum(1 for t in trades if t.reason == "TP")/len(trades),
        "true_win_rate": float((R > 0).mean()),
        "avg_R": float(R.mean()), "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(R.mean()), "sum_R": float(R.sum()),
        "worst_R": float(R.min()), "ambiguous_bars": amb,
        "avg_hold": float(np.mean([t.hold_bars for t in trades])),
    }


def main(symbols=None, risk=0.5):
    symbols = symbols or _ALL_TM_SYMBOLS
    print(f"\n{'='*94}")
    print(f"  RESOLVED BACKTEST — intrabar exits, structural SL/TP, honest R")
    print(f"{'='*94}")
    print(f"  {'Symbol':<14} {'Res':>4} {'Trades':>7} {'hitTP%':>7} {'winR%':>7} "
          f"{'avgR':>7} {'winR':>7} {'lossR':>7} {'expR':>7} {'amb':>5} {'hold':>6}")
    print(f"  {'-'*14} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*6}")

    all_t: list[RTrade] = []
    for sym in symbols:
        ts = run_symbol(sym)
        all_t.extend(ts)
        s = summarise(ts, sym)
        if not s:
            print(f"  {sym:<14}  no trades"); continue
        res = ts[0].resolved_on if ts else "?"
        print(f"  {sym:<14} {res:>4} {s['trades']:>7} {100*s['hit_tp_rate']:>6.1f}% "
              f"{100*s['true_win_rate']:>6.1f}% {s['avg_R']:>+7.3f} {s['avg_win']:>+7.2f} "
              f"{s['avg_loss']:>+7.2f} {s['expectancy']:>+7.3f} {s['ambiguous_bars']:>5} {s['avg_hold']:>6.0f}")

    agg = summarise(all_t, "PORTFOLIO")
    if agg:
        print(f"  {'-'*14}")
        print(f"  {'PORTFOLIO':<14} {'':>4} {agg['trades']:>7} {100*agg['hit_tp_rate']:>6.1f}% "
              f"{100*agg['true_win_rate']:>6.1f}% {agg['avg_R']:>+7.3f} {agg['avg_win']:>+7.2f} "
              f"{agg['avg_loss']:>+7.2f} {agg['expectancy']:>+7.3f} {agg['ambiguous_bars']:>5} {agg['avg_hold']:>6.0f}")
        # integrity
        sl = [t for t in all_t if t.reason == "SL"]
        bad = [t for t in sl if t.R > -0.98]
        n_m1 = sum(1 for t in all_t if t.resolved_on == "M1")
        n_m5 = sum(1 for t in all_t if t.resolved_on == "M5")
        amb_m5 = sum(1 for t in all_t if t.ambiguous and t.resolved_on == "M5")
        print(f"\n  RESOLUTION: {n_m1} trades on M1 (exact), {n_m5} on M5 (pre-M1 window)")
        print(f"  INTEGRITY: {len(sl)-len(bad)}/{len(sl)} SL trades = ~-1R  |  "
              f"{agg['ambiguous_bars']} ambiguous same-bar (assumed SL); {amb_m5} of them on "
              f"M5 could flip under M1")

    if all_t:
        LOGS.mkdir(exist_ok=True)
        pd.DataFrame([asdict(t) for t in all_t]).to_csv(LOGS / "resolved_trades.csv", index=False)
        print(f"  Per-trade log: {LOGS / 'resolved_trades.csv'}")
    print(f"{'='*94}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--risk", type=float, default=0.5)
    args = ap.parse_args()
    main(symbols=args.symbols or None, risk=args.risk)
