"""Systematic adaptive-risk parameter testing harness.

Not a casual dial-nudge — a structured sweep over every risk knob, scored against
TWO objectives (Phase-1 and funded are different problems):

  Phase-1  objective = pass_pct                 (hit +10% in 30 days)
  Funded   objective = survival = 100 - blow - daily
                                                (no deadline; only breaches lose
                                                 the account, so timeouts are fine)

Stage 1 (this file): one-at-a-time sensitivity around a baseline — find which
knobs actually move each objective, at enough MC trials that the ranking is real.
Every row is logged to logs/adaptive_sweep.csv for stage-2 joint optimization.

Data is loaded + signalled ONCE and reused across all runs.

Usage
-----
    python -m backtests.sweep_adaptive --mc 800
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 console safety
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.run_ftmo_sim import INSTRUMENTS
from backtests.run_ftmo_barsim import _load_symbol, _build_events, _run

OUT = Path(__file__).resolve().parent.parent / "logs" / "adaptive_sweep.csv"

# Baseline config (informed by the buffer sweep: 0.5 is a sweet spot).
BASELINE = dict(base_risk=1.0, cap=4.0, guard_buffer=0.5, score_edge=1, min_trade=0.25)

# Every knob to test, one at a time, with its candidate grid.
GRID = {
    "base_risk":    [0.5, 0.75, 1.0, 1.25, 1.5],
    "cap":          [3.0, 4.0, 6.0, 8.0],
    "guard_buffer": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "score_edge":   [0, 1, 2, 3],
    "min_trade":    [0.1, 0.25, 0.5, 0.75],
}


def _score(res):
    phase1 = res["pass_pct"]                               # maximize
    funded = 100.0 - res["blow_pct"] - res["daily_pct"]    # maximize survival
    return phase1, funded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=int, default=4)
    ap.add_argument("--mc",    type=int, default=800)
    args = ap.parse_args()

    print("Loading + signalling instruments (once)...")
    data = {}
    for sym, cfg in INSTRUMENTS.items():
        d = _load_symbol(sym, cfg, args.score)
        if d is not None:
            data[sym] = d
    events = _build_events(data)
    print(f"  {len(data)} instruments, {len(events)} bar-events, mc={args.mc}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = ["param", "value", "pass", "blow", "daily", "timeout",
              "avg_trades", "phase1_obj", "funded_obj"]

    def run_one(param, value, cfg):
        r = _run(events, data, cfg["base_risk"], cfg["cap"], True, args.mc,
                 random.Random(7),                     # fixed seed → paired windows
                 guard_buffer=cfg["guard_buffer"],
                 score_edge=cfg["score_edge"], min_trade=cfg["min_trade"])
        p1, fu = _score(r)
        row = dict(param=param, value=value, pass_=r["pass_pct"], blow=r["blow_pct"],
                   daily=r["daily_pct"], timeout=r["timeout_pct"],
                   avg_trades=r["avg_trades"], phase1_obj=p1, funded_obj=fu)
        rows.append(row)
        print(f"  {param:<13}={str(value):<6}  pass {r['pass_pct']:5.1f}  "
              f"blow {r['blow_pct']:4.2f}  daily {r['daily_pct']:4.1f}  "
              f"timeout {r['timeout_pct']:5.1f}  trades {r['avg_trades']:4.1f}  "
              f"| P1 {p1:5.1f}  FUND {fu:5.1f}")
        return row

    # Baseline
    print("BASELINE:", BASELINE)
    run_one("baseline", "-", dict(BASELINE))

    # One-at-a-time sensitivity
    for param, values in GRID.items():
        print(f"\n-- {param} --")
        for v in values:
            cfg = dict(BASELINE)
            cfg[param] = v
            run_one(param, v, cfg)

    # Persist
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r["param"], r["value"], f"{r['pass_']:.2f}", f"{r['blow']:.2f}",
                        f"{r['daily']:.2f}", f"{r['timeout']:.2f}", f"{r['avg_trades']:.2f}",
                        f"{r['phase1_obj']:.2f}", f"{r['funded_obj']:.2f}"])

    # Best per objective
    best_p1 = max(rows, key=lambda r: r["phase1_obj"])
    best_fu = max(rows, key=lambda r: r["funded_obj"])
    print(f"\n{'='*66}")
    print(f"  BEST Phase-1 (pass): {best_p1['param']}={best_p1['value']} → "
          f"pass {best_p1['pass_']:.1f}%  blow {best_p1['blow']:.2f}%")
    print(f"  BEST Funded (survive): {best_fu['param']}={best_fu['value']} → "
          f"survival {best_fu['funded_obj']:.1f}%  (blow {best_fu['blow']:.2f}%)")
    print(f"  Full grid → {OUT}")
    print("="*66)
    print("  NOTE: stage-1 one-at-a-time sensitivity. High-impact knobs go to a")
    print("  stage-2 JOINT sweep before any value is trusted for live.")


if __name__ == "__main__":
    main()
