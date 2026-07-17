"""Stage-2 JOINT adaptive-risk sweep — base_risk x guard_buffer together.

Stage-1 (sweep_adaptive.py) was one-at-a-time and found these two as the movers.
They interact (risk sets pass/blow; buffer sets how the daily guard trades daily
breaches for timeouts), so the final live values must be chosen JOINTLY.

Two objectives, reported separately:
  Phase-1 = max pass_pct
  Funded  = max survival (100 - blow - daily)

Grid logged to logs/joint_sweep.csv. Data loaded + signalled once.

Usage:  python -m backtests.sweep_joint --mc 600
"""
from __future__ import annotations

import argparse, csv, random, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtests.run_ftmo_sim import INSTRUMENTS
from backtests.run_ftmo_barsim import _load_symbol, _build_events, _run

OUT = Path(__file__).resolve().parent.parent / "logs" / "joint_sweep.csv"

RISKS   = [0.75, 1.0, 1.25, 1.5]
BUFFERS = [0.25, 0.5, 0.75, 1.0]
CAP, SCORE_EDGE, MIN_TRADE = 4.0, 1, 0.5   # stage-1 secondary optima, held fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=int, default=4)
    ap.add_argument("--mc",    type=int, default=600)
    a = ap.parse_args()

    print("Loading + signalling (once)...")
    data = {s: d for s, cfg in INSTRUMENTS.items()
            if (d := _load_symbol(s, cfg, a.score)) is not None}
    events = _build_events(data)
    print(f"  {len(data)} instruments, {len(events)} events, mc={a.mc}\n")
    print(f"  {'risk':>5} {'buf':>5} | {'pass':>6} {'blow':>6} {'daily':>6} "
          f"{'timeout':>8} {'trades':>7} | {'P1':>6} {'FUND':>6}")
    print("  " + "-" * 74)

    rows = []
    for r in RISKS:
        for b in BUFFERS:
            res = _run(events, data, r, CAP, True, a.mc, random.Random(7),
                       guard_buffer=b, score_edge=SCORE_EDGE, min_trade=MIN_TRADE)
            p1 = res["pass_pct"]
            fu = 100.0 - res["blow_pct"] - res["daily_pct"]
            rows.append(dict(risk=r, buf=b, **res, p1=p1, fu=fu))
            print(f"  {r:>5} {b:>5} | {res['pass_pct']:6.1f} {res['blow_pct']:6.2f} "
                  f"{res['daily_pct']:6.2f} {res['timeout_pct']:8.1f} "
                  f"{res['avg_trades']:7.1f} | {p1:6.1f} {fu:6.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["risk", "buf", "pass", "blow", "daily", "timeout", "avg_trades", "phase1", "funded"])
        for x in rows:
            w.writerow([x["risk"], x["buf"], f"{x['pass_pct']:.2f}", f"{x['blow_pct']:.2f}",
                        f"{x['daily_pct']:.2f}", f"{x['timeout_pct']:.2f}",
                        f"{x['avg_trades']:.2f}", f"{x['p1']:.2f}", f"{x['fu']:.2f}"])

    bp1 = max(rows, key=lambda x: x["p1"])
    bfu = max(rows, key=lambda x: (x["fu"], x["p1"]))   # tie-break survival on pass
    print("\n" + "=" * 76)
    print(f"  PHASE-1 config:  risk={bp1['risk']}  buffer={bp1['buf']}  "
          f"-> pass {bp1['pass_pct']:.1f}%  blow {bp1['blow_pct']:.2f}%  daily {bp1['daily_pct']:.2f}%")
    print(f"  FUNDED  config:  risk={bfu['risk']}  buffer={bfu['buf']}  "
          f"-> survival {bfu['fu']:.1f}%  (blow {bfu['blow_pct']:.2f}%, pass {bfu['pass_pct']:.1f}%)")
    print(f"  grid -> {OUT}")
    print("=" * 76)


if __name__ == "__main__":
    main()
