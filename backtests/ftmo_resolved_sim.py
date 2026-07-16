"""FTMO prop-firm sim on the HONEST resolved pool — spread, daily halt, heat.

Compounds everything onto the intrabar-resolved trade log (logs/resolved_trades.csv):
  - Spread cost: each trade's R is haircut by (typical_spread / structural_rd) —
    tight structural stops are spread-sensitive, so this matters a lot.
  - Size: the analyzer's grade-based size_mult scales each trade's risk.
  - Portfolio: shared equity, live daily logic — halt new entries at -2%, hard
    close-all at -5% daily / -10% total. Days bootstrapped from real entry-day
    batches (preserves loser clustering).
  - Portfolio heat: max concurrent open risk from real entry/exit windows.

Reports pass / blow (daily vs total) / payout / days-to-pass per risk level, for
the full universe AND a viable-only universe (net-positive symbols after spread).

Usage:
    python -m backtests.ftmo_resolved_sim
    python -m backtests.ftmo_resolved_sim --runs 20000 --drop XAGUSD UK100.cash
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOGS = Path(__file__).resolve().parent.parent / "logs"

# Live per-symbol spread in PRICE (from MT5 2026-07-16)
SPREAD = {
    "XAUUSD": 0.47, "XAGUSD": 0.047, "GBPUSD": 4e-05, "US100.cash": 1.93,
    "US30.cash": 2.63, "US500.cash": 0.60, "US2000.cash": 0.96,
    "UK100.cash": 5.41, "JP225.cash": 10.0, "GER40.cash": 3.39,
}

# FTMO rules
TARGET_P1, TARGET_P2 = 0.10, 0.05
DAILY_HALT  = 0.02     # stop new entries
DAILY_FLOOR = 0.05     # hard close-all
TOTAL_FLOOR = 0.10
MIN_DAYS, TIME_CAP = 4, 60
DEFAULT_RISK = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def load_pool(drop=None):
    df = pd.read_csv(LOGS / "resolved_trades.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    if drop:
        df = df[~df["symbol"].isin(drop)]
    df["rd"] = (df["entry"] - df["stop"]).abs()
    df["spread_R"] = df["symbol"].map(SPREAD) / df["rd"]
    # net, size-weighted R in units of BASE risk (size_mult scales risk taken)
    df["eff_R"] = (df["R"] - df["spread_R"]) * df["size_mult"]
    df["day"] = df["entry_time"].dt.date
    return df


def daily_batches(df):
    g = defaultdict(list)
    for _, r in df.sort_values("entry_time").iterrows():
        g[str(r["day"])].append(float(r["eff_R"]))
    return [g[d] for d in sorted(g.keys())]


def portfolio_heat(df, risk_pct):
    """Max concurrent open risk (% of account) from real entry/exit windows."""
    events = []
    for _, r in df.iterrows():
        events.append((r["entry_time"], +risk_pct * r["size_mult"]))
        xt = r["exit_time"] if pd.notna(r["exit_time"]) else r["entry_time"]
        events.append((xt, -risk_pct * r["size_mult"]))
    events.sort(key=lambda e: e[0])
    heat = peak = 0.0
    for _, d in events:
        heat += d
        peak = max(peak, heat)
    return peak


def sim_phase(rng, batches, risk_frac, target):
    n = len(batches); eq = 1.0; floor = 1 - TOTAL_FLOOR; tgt = 1 + target
    for day in range(TIME_CAP):
        batch = batches[rng.integers(n)]
        day_start = eq; dfloor = day_start * (1 - DAILY_FLOOR)
        halt = day_start * (1 - DAILY_HALT)
        for R in batch:
            eq *= (1 + R * risk_frac)
            if eq <= floor: return "blown_total", day + 1
            if eq <= dfloor: return "blown_daily", day + 1
            if eq <= halt: break            # daily halt: no more new entries today
        if eq >= tgt and day + 1 >= MIN_DAYS:
            return "pass", day + 1
    return "timeout", TIME_CAP


def run_level(batches, risk_pct, runs, seed=42):
    rng = np.random.default_rng(seed); frac = risk_pct / 100
    p1 = blow_d = blow_t = full = 0; days = []
    for _ in range(runs):
        o1, d1 = sim_phase(rng, batches, frac, TARGET_P1)
        if o1 == "pass":
            p1 += 1; days.append(d1)
            o2, _ = sim_phase(rng, batches, frac, TARGET_P2)
            if o2 == "pass": full += 1
        elif o1 == "blown_daily": blow_d += 1
        elif o1 == "blown_total": blow_t += 1
    return dict(risk=risk_pct, pass_rate=p1/runs, blow=(blow_d+blow_t)/runs,
                blow_daily=blow_d/runs, blow_total=blow_t/runs,
                full=full/runs, med_days=float(np.median(days)) if days else float("nan"))


def report(df, label, risk_levels, runs):
    batches = daily_batches(df)
    allR = np.array([r for b in batches for r in b])
    print(f"\n{'='*92}")
    print(f"  {label}")
    print(f"  {df['symbol'].nunique()} symbols, {len(df)} trades, {len(batches)} days, "
          f"net expectancy {allR.mean():+.3f}R (spread-adjusted), "
          f"median {np.median([len(b) for b in batches]):.0f} trades/day")
    print(f"{'='*92}")
    print(f"  {'Risk%':>6} {'pass':>7} {'BLOW':>7} {'daily':>7} {'total':>7} "
          f"{'full':>7} {'payout*':>8} {'medDays':>8} {'heat%':>7}")
    rows = []
    for rp in risk_levels:
        m = run_level(batches, rp, runs)
        heat = portfolio_heat(df, rp)
        payout = m["full"] * 0.9        # ~90% of funded reach first payout (proxy)
        rows.append({**m, "heat": heat, "label": label})
        print(f"  {rp:>5.2f}% {100*m['pass_rate']:>6.1f}% {100*m['blow']:>6.1f}% "
              f"{100*m['blow_daily']:>6.1f}% {100*m['blow_total']:>6.1f}% "
              f"{100*m['full']:>6.1f}% {100*payout:>7.1f}% {m['med_days']:>8.0f} {heat:>6.1f}%")
    return rows


def main(runs=20000, drop_default=("XAGUSD", "UK100.cash"), risk_levels=None):
    risk_levels = risk_levels or DEFAULT_RISK
    full_df = load_pool()
    viable_df = load_pool(drop=list(drop_default))

    r1 = report(full_df, "FULL UNIVERSE (all 10 symbols, spread-adjusted)", risk_levels, runs)
    r2 = report(viable_df, f"VIABLE UNIVERSE (dropped {', '.join(drop_default)})", risk_levels, runs)

    out = pd.DataFrame(r1 + r2)
    out.to_csv(LOGS / "ftmo_resolved_sim.csv", index=False)
    print(f"\n  Data: {LOGS / 'ftmo_resolved_sim.csv'}")
    print("  *payout = full-eval pass x 0.9 (funded reaching first payout, proxy)")
    _plot(r1, r2, drop_default)


def _plot(r1, r2, dropped):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    d1 = pd.DataFrame(r1); d2 = pd.DataFrame(r2)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6)); fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("FTMO on honest resolved pool — full vs viable universe (spread-adjusted)",
                 color="#fff", fontweight="bold")
    for a in ax:
        a.set_facecolor("#1a1a1a"); a.tick_params(colors="#ccc"); a.grid(color="#2a2a2a", lw=0.4)
        for s in a.spines.values(): s.set_color("#333")
        for l in a.get_xticklabels()+a.get_yticklabels(): l.set_color("#ccc")
    ax[0].set_title("Blow-up rate", color="#ccc")
    ax[0].plot(d1.risk, 100*d1.blow, "-o", color="#ff6644", label="full")
    ax[0].plot(d2.risk, 100*d2.blow, "-o", color="#44ff88", label=f"viable (-{len(dropped)})")
    ax[1].set_title("Full-eval pass rate", color="#ccc")
    ax[1].plot(d1.risk, 100*d1.full, "-o", color="#ff6644", label="full")
    ax[1].plot(d2.risk, 100*d2.full, "-o", color="#44ff88", label="viable")
    for a in ax:
        a.set_xlabel("Risk per trade %", color="#ccc"); a.legend(facecolor="#1a1a1a", labelcolor="#ccc")
    plt.tight_layout(rect=[0,0,1,0.95])
    p = LOGS / "ftmo_resolved_sim.png"; plt.savefig(p, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"  Chart: {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--drop", nargs="*", default=["XAGUSD", "UK100.cash"])
    args = ap.parse_args()
    main(runs=args.runs, drop_default=tuple(args.drop))
