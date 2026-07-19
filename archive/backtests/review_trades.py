"""Retrospective post-trade review of trades already taken.

Reads logs/trades.jsonl and, for each closed trade, reconstructs the M15 context
at entry and runs the TradeAnalyzer on it:
  - analyze_entry(): what the liquidity system WOULD have graded this trade
    (type, grade, structural stop, real liquidity target) — vs what was actually
    placed (often an old arithmetic-TP / ATR-stop trade).
  - analyze_exit(): the post-mortem — hit target?, MFE/MAE in R, thesis valid,
    the lesson.

Writes the enriched records to logs/trades_reviewed.jsonl and prints a breakdown.

Usage:
    python -m backtests.review_trades
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.trade_analyzer import analyze_entry, analyze_exit

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LOGS      = ROOT / "logs"


def _ctx(symbol: str, open_time: str, close_time: str):
    """Return (context_df_up_to_entry, atr, path_high, path_low) or (None,...)."""
    p = PROCESSED / f"{symbol}_M15.csv"
    if not p.exists():
        return None, 0.0, None, None
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    ot = pd.Timestamp(open_time)
    ct = pd.Timestamp(close_time) if close_time else ot
    ctx = df[df["time"] <= ot].tail(120).reset_index(drop=True)
    if len(ctx) < 60:
        return None, 0.0, None, None
    atr = float((ctx["high"] - ctx["low"]).rolling(14).mean().iloc[-1])
    span = df[(df["time"] >= ot) & (df["time"] <= ct)]
    path_high = float(span["high"].max()) if len(span) else None
    path_low  = float(span["low"].min())  if len(span) else None
    return ctx, atr, path_high, path_low


def main() -> None:
    src = LOGS / "trades.jsonl"
    if not src.exists():
        print("No trades.jsonl to review.")
        return
    trades = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not trades:
        print("Journal is empty.")
        return

    print(f"\n{'='*90}")
    print(f"  RETROSPECTIVE TRADE REVIEW — {len(trades)} trade(s)")
    print(f"{'='*90}")

    reviewed = []
    for t in trades:
        sym  = t["symbol"]
        dirn = t["direction"]
        entry = t["entry_price"]
        sl    = t["sl_price"]
        close = t.get("close_price", entry)

        ctx, atr, path_high, path_low = _ctx(sym, t["open_time"], t.get("close_time"))
        if path_high is None:
            path_high = max(entry, close)
        if path_low is None:
            path_low = min(entry, close)

        actual_rd  = abs(entry - sl)
        actual_rr  = abs((t.get("tp_price") or entry) - entry) / actual_rd if actual_rd else 0

        print(f"\n  {sym} {'LONG' if dirn==1 else 'SHORT'}  entry={entry:.2f}")
        print(f"    ACTUAL (as placed): SL={sl:.2f} (rd {actual_rd:.1f})  "
              f"TP={t.get('tp_price')}  RR~{actual_rr:.2f}  "
              f"outcome={t.get('outcome')}  R={t.get('r_multiple',0):+.2f}")

        # What the liquidity system would have done
        if ctx is not None:
            plan = analyze_entry(df_m15=ctx, df_m5=None, direction=dirn,
                                 entry=entry, stop=sl, atr=atr, h4_bias=dirn)
            print(f"    ANALYZER would say: grade={plan.grade} type={plan.trade_type} "
                  f"stop@{plan.stop_src} {plan.stop:.2f} -> target {plan.tp:.2f} "
                  f"RR={plan.rr} size={plan.size_mult}x tradeable={plan.tradeable}")
            print(f"      thesis: {plan.thesis}")
            target = plan.tp
        else:
            print(f"    ANALYZER: no M15 context for {sym} at that time — skipped")
            plan = None
            target = t.get("tp_price") or entry

        # Post-mortem on the actual outcome
        review = analyze_exit(direction=dirn, entry=entry, stop=sl, target=target,
                              exit_px=close, path_high=path_high, path_low=path_low,
                              reason=(t.get("outcome") or "").upper())
        print(f"    POST-MORTEM: hitTP={review.hit_target}  "
              f"MFE={review.mfe_R:+.2f}R  MAE={review.mae_R:+.2f}R  "
              f"thesis_valid={review.thesis_valid}")
        print(f"      lesson: {review.lesson}")
        for n in review.notes:
            print(f"      - {n}")

        rec = dict(t)
        if plan is not None:
            rec["analyzer_grade"] = plan.grade
            rec["analyzer_type"]  = plan.trade_type
            rec["analyzer_stop"]  = plan.stop
            rec["analyzer_target"] = plan.tp
            rec["analyzer_rr"]    = plan.rr
            rec["analyzer_tradeable"] = plan.tradeable
        rec["review_hit_target"]  = review.hit_target
        rec["review_mfe_r"]       = review.mfe_R
        rec["review_mae_r"]       = review.mae_R
        rec["review_lesson"]      = review.lesson
        rec["review_notes"]       = review.notes
        reviewed.append(rec)

    out = LOGS / "trades_reviewed.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in reviewed:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Reviewed journal written: {out}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
