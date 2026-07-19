"""FTMO Monte Carlo with protections — daily halt + same-direction correlation cap.

Layers two defences onto the honest-pool challenge sim and measures each one's
contribution to blow-rate:

  BASELINE      : no intraday brake (worst case — what ftmo_montecarlo.py showed)
  +DAILY HALT   : stop taking new trades once the day is down -halt_pct (the live
                  orchestrator's MAX_DAILY_LOSS_PCT=2.0 brake — the MC previously ignored it)
  +CORR CAP     : also drop the (N+1)th concurrent SAME-DIRECTION trade within a
                  correlation group on a given day — attacks loser clustering directly

The correlation cap is modelled on the real per-trade log (symbol + direction +
day), so it reflects trades that genuinely would have been blocked.

Usage:
    python -m backtests.ftmo_mc_protected
    python -m backtests.ftmo_mc_protected --halt 2.0 --cap 3 --runs 20000
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.ftmo_montecarlo import (
    PROFIT_TARGET_P1, PROFIT_TARGET_P2, MAX_TOTAL_LOSS, MAX_DAILY_LOSS,
    MIN_TRADING_DAYS, TIME_CAP_DAYS,
)

LOGS = Path(__file__).resolve().parent.parent / "logs"

# Correlation groups — same as execution/orchestrator.py _CORR_GROUPS
CORR_GROUP = {
    "US30.cash": "US", "US100.cash": "US", "US500.cash": "US", "US2000.cash": "US",
    "UK100.cash": "EU", "GER40.cash": "EU",
    "XAUUSD": "METAL", "XAGUSD": "METAL",
    "GBPUSD": "FX",
}

DEFAULT_RISK = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def _load_df() -> pd.DataFrame:
    csv = LOGS / "audit_naive_trades.csv"
    df = pd.read_csv(csv)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["day"] = df["entry_time"].dt.date        # cluster by ENTRY day (concurrency proxy)
    df["group"] = df["symbol"].map(CORR_GROUP).fillna("SOLO")
    return df.sort_values("entry_time").reset_index(drop=True)


def build_batches(df: pd.DataFrame, corr_cap: int | None = None) -> list[list[float]]:
    """Group R-outcomes into per-day batches. If corr_cap set, drop the (cap+1)th
    same-direction trade within a correlation group on the same entry-day."""
    batches = []
    for day, day_df in df.groupby("day"):
        if corr_cap is not None:
            keep_idx = []
            counters: dict = defaultdict(int)
            # chronological within the day
            for idx, row in day_df.sort_values("entry_time").iterrows():
                key = (row["group"], row["direction"])
                counters[key] += 1
                if counters[key] <= corr_cap:
                    keep_idx.append(idx)
                # else: this concurrent same-dir correlated trade is blocked
            day_df = day_df.loc[keep_idx]
        rs = [float(r) for r in day_df["R"].values]
        if rs:
            batches.append(rs)
    return batches


def simulate_phase(rng, batches, risk_frac, target, halt_frac=None):
    """One FTMO phase. halt_frac: stop new trades once day down this fraction (e.g. 0.02)."""
    n_days   = len(batches)
    equity   = 1.0
    floor    = 1.0 - MAX_TOTAL_LOSS
    target_e = 1.0 + target

    for day in range(TIME_CAP_DAYS):
        batch     = batches[rng.integers(n_days)]
        day_start = equity
        daily_floor = day_start * (1.0 - MAX_DAILY_LOSS)
        halt_level  = day_start * (1.0 - halt_frac) if halt_frac else None

        for R in batch:
            equity *= (1.0 + R * risk_frac)
            if equity <= floor:
                return "blown_total", day + 1
            if equity <= daily_floor:
                return "blown_daily", day + 1
            # daily halt: stop opening new trades for the rest of the day
            if halt_level is not None and equity <= halt_level:
                break

        if equity >= target_e and (day + 1) >= MIN_TRADING_DAYS:
            return "pass", day + 1

    return "timeout", TIME_CAP_DAYS


def run_config(batches, risk_pct, runs, halt_frac=None, seed=42):
    rng  = np.random.default_rng(seed)
    frac = risk_pct / 100.0
    n_pass = n_blow = n_full = 0
    days_p = []
    for _ in range(runs):
        o1, d1 = simulate_phase(rng, batches, frac, PROFIT_TARGET_P1, halt_frac)
        if o1 == "pass":
            n_pass += 1; days_p.append(d1)
            o2, _ = simulate_phase(rng, batches, frac, PROFIT_TARGET_P2, halt_frac)
            if o2 == "pass":
                n_full += 1
        elif o1.startswith("blown"):
            n_blow += 1
    return dict(
        risk_pct=risk_pct,
        pass_rate=n_pass/runs, blow_rate=n_blow/runs, full_pass=n_full/runs,
        med_days=float(np.median(days_p)) if days_p else float("nan"),
    )


def main(risk_levels=None, halt_pct=2.0, corr_cap=3, runs=20_000):
    risk_levels = risk_levels or DEFAULT_RISK
    df = _load_df()

    base_batches = build_batches(df, corr_cap=None)
    cap_batches  = build_batches(df, corr_cap=corr_cap)

    n_before = sum(len(b) for b in base_batches)
    n_after  = sum(len(b) for b in cap_batches)
    print(f"\n{'='*104}")
    print(f"  FTMO MONTE CARLO — PROTECTION LAYERS  ({runs:,} runs/level)")
    print(f"  Daily halt: -{halt_pct}%   |   Corr cap: max {corr_cap} concurrent same-dir per group")
    print(f"  Corr cap removes {n_before - n_after}/{n_before} trades "
          f"({100*(n_before-n_after)/n_before:.1f}% — the clustered correlated ones)")
    print(f"{'='*104}")

    halt_frac = halt_pct / 100.0
    configs = [
        ("BASELINE (no brake)",      base_batches, None),
        (f"+DAILY HALT -{halt_pct}%", base_batches, halt_frac),
        (f"+HALT +CORR CAP {corr_cap}", cap_batches, halt_frac),
    ]

    all_rows = []
    for label, batches, hf in configs:
        print(f"\n  --- {label} ---")
        print(f"  {'Risk%':>6} {'pass':>7} {'BLOW':>7} {'full':>7} {'medDays':>8}")
        print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
        for rp in risk_levels:
            m = run_config(batches, rp, runs, halt_frac=hf)
            m["config"] = label
            all_rows.append(m)
            print(f"  {rp:>5.2f}% {100*m['pass_rate']:>6.1f}% {100*m['blow_rate']:>6.1f}% "
                  f"{100*m['full_pass']:>6.1f}% {m['med_days']:>8.0f}")

    out = pd.DataFrame(all_rows)
    out.to_csv(LOGS / "ftmo_mc_protected.csv", index=False)
    print(f"\n  Data saved: {LOGS / 'ftmo_mc_protected.csv'}")

    # Blow-rate reduction summary at each risk level
    print(f"\n  === BLOW-RATE REDUCTION (baseline -> +halt -> +cap) ===")
    print(f"  {'Risk%':>6} {'baseline':>9} {'+halt':>9} {'+halt+cap':>11} {'total cut':>10}")
    piv = out.pivot_table(index="risk_pct", columns="config", values="blow_rate")
    labels = [c[0] for c in configs]
    for rp in risk_levels:
        b  = 100*piv.loc[rp, labels[0]]
        h  = 100*piv.loc[rp, labels[1]]
        c  = 100*piv.loc[rp, labels[2]]
        print(f"  {rp:>5.2f}% {b:>8.1f}% {h:>8.1f}% {c:>10.1f}% {b-c:>9.1f}pp")

    _plot(out, configs, risk_levels, halt_pct, corr_cap)
    print(f"{'='*104}\n")


def _plot(out, configs, risk_levels, halt_pct, corr_cap):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [c[0] for c in configs]
    colors = ["#ff4444", "#ffaa00", "#44ff88"]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(f"Protection layers: -{halt_pct}% daily halt + max-{corr_cap} same-dir correlation cap",
                 color="#ffffff", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#cccccc")
        ax.grid(color="#2a2a2a", linewidth=0.4)
        for s in ax.spines.values(): s.set_color("#333333")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_color("#cccccc")

    piv_blow = out.pivot_table(index="risk_pct", columns="config", values="blow_rate")
    piv_full = out.pivot_table(index="risk_pct", columns="config", values="full_pass")
    x = list(piv_blow.index)

    ax = axes[0]
    ax.set_title("Blow-up rate — each layer's contribution", color="#cccccc")
    for lab, col in zip(labels, colors):
        ax.plot(x, 100*piv_blow[lab], "-o", color=col, label=lab, linewidth=2)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("Blow-up rate %", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    ax = axes[1]
    ax.set_title("Full-eval pass rate — protection keeps the edge", color="#cccccc")
    for lab, col in zip(labels, colors):
        ax.plot(x, 100*piv_full[lab], "-o", color=col, label=lab, linewidth=2)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("Full pass rate %", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    o = LOGS / "ftmo_mc_protected.png"
    plt.savefig(o, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved: {o}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, nargs="+", default=None)
    ap.add_argument("--halt", type=float, default=2.0)
    ap.add_argument("--cap",  type=int,   default=3)
    ap.add_argument("--runs", type=int,   default=20_000)
    args = ap.parse_args()
    main(risk_levels=args.risk, halt_pct=args.halt, corr_cap=args.cap, runs=args.runs)
