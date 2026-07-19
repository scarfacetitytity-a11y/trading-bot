"""FTMO blow-up autopsy — WHY do accounts die, not just how many.

Instruments the Monte Carlo to record every death: was it the daily -5% rule or
the total -10% floor? On which day? How deep into a session? And critically —
what did the FATAL day look like: how many trades fired, how many lost together?

The shared-account structure fires ~5 trades/day across correlated instruments.
At 38% win rate a cluster of simultaneous losers is the real killer, not a single
bad trade. This script proves whether that's the mechanism.

Usage:
    python -m backtests.ftmo_blowup_analysis
    python -m backtests.ftmo_blowup_analysis --risk 0.5 1.0 2.0 --runs 20000
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
    load_daily_batches, PROFIT_TARGET_P1, MAX_TOTAL_LOSS, MAX_DAILY_LOSS,
    MIN_TRADING_DAYS, TIME_CAP_DAYS,
)

LOGS = Path(__file__).resolve().parent.parent / "logs"


def simulate_instrumented(rng, daily_batches, risk_frac, target=PROFIT_TARGET_P1):
    """Run one P1 challenge, returning a full death/pass record."""
    n_days   = len(daily_batches)
    equity   = 1.0
    floor    = 1.0 - MAX_TOTAL_LOSS
    target_e = 1.0 + target
    peak     = 1.0

    for day in range(TIME_CAP_DAYS):
        batch     = daily_batches[rng.integers(n_days)]
        day_start = equity
        daily_floor = day_start * (1.0 - MAX_DAILY_LOSS)

        n_trades = len(batch)
        n_losers = sum(1 for R in batch if R <= 0)
        day_R    = 0.0

        for R in batch:
            equity *= (1.0 + R * risk_frac)
            day_R  += R
            peak    = max(peak, equity)
            if equity <= floor:
                return dict(outcome="blown_total", day=day + 1, equity=equity,
                            fatal_day_R=day_R, fatal_trades=n_trades,
                            fatal_losers=n_losers, peak=peak,
                            gave_back=peak - equity)
            if equity <= daily_floor:
                return dict(outcome="blown_daily", day=day + 1, equity=equity,
                            fatal_day_R=day_R, fatal_trades=n_trades,
                            fatal_losers=n_losers, peak=peak,
                            gave_back=peak - equity)

        if equity >= target_e and (day + 1) >= MIN_TRADING_DAYS:
            return dict(outcome="pass", day=day + 1, equity=equity, peak=peak)

    return dict(outcome="timeout", day=TIME_CAP_DAYS, equity=equity, peak=peak)


def analyse_risk(daily_batches, risk_pct, runs, seed=42):
    rng   = np.random.default_rng(seed)
    frac  = risk_pct / 100.0
    deaths = []
    outcomes = defaultdict(int)

    for _ in range(runs):
        rec = simulate_instrumented(rng, daily_batches, frac)
        outcomes[rec["outcome"]] += 1
        if rec["outcome"].startswith("blown"):
            deaths.append(rec)

    total_blown = outcomes["blown_daily"] + outcomes["blown_total"]
    d_daily = [d for d in deaths if d["outcome"] == "blown_daily"]
    d_total = [d for d in deaths if d["outcome"] == "blown_total"]

    def _med(vals, key):
        arr = [d[key] for d in vals]
        return float(np.median(arr)) if arr else float("nan")

    return dict(
        risk_pct=risk_pct,
        pass_rate=outcomes["pass"] / runs,
        blow_rate=total_blown / runs,
        blown_daily=len(d_daily),
        blown_total=len(d_total),
        pct_deaths_daily=(len(d_daily) / total_blown) if total_blown else 0.0,
        pct_deaths_total=(len(d_total) / total_blown) if total_blown else 0.0,
        med_day_of_death=_med(deaths, "day"),
        med_fatal_losers=_med(d_daily, "fatal_losers"),
        med_fatal_trades=_med(d_daily, "fatal_trades"),
        med_fatal_day_R=_med(d_daily, "fatal_day_R"),
        med_gaveback_total=_med(d_total, "gave_back"),
    )


def analyse_daily_distribution(daily_batches):
    """Structural analysis of the historical day pool — the raw risk material."""
    trades_per_day = np.array([len(d) for d in daily_batches])
    day_R_sum      = np.array([sum(d) for d in daily_batches])
    losers_per_day = np.array([sum(1 for R in d if R <= 0) for d in daily_batches])
    # worst realistic day: sum of only the losing R that day
    loss_only_R    = np.array([sum(R for R in d if R <= 0) for d in daily_batches])

    print(f"\n  === DAILY POOL STRUCTURE ({len(daily_batches)} historical days) ===")
    print(f"  Trades/day    : median {np.median(trades_per_day):.0f}, "
          f"mean {trades_per_day.mean():.1f}, max {trades_per_day.max()}")
    print(f"  Day net R      : median {np.median(day_R_sum):+.2f}, "
          f"p5 {np.percentile(day_R_sum,5):+.2f}, min {day_R_sum.min():+.2f}")
    print(f"  Loss-only R/day: median {np.median(loss_only_R):+.2f}, "
          f"p5 {np.percentile(loss_only_R,5):+.2f}, worst {loss_only_R.min():+.2f}")
    print(f"  Losers/day     : median {np.median(losers_per_day):.0f}, "
          f"max {losers_per_day.max()}")

    # What does the worst 5% of days cost at each risk level?
    p5_loss = np.percentile(loss_only_R, 5)     # e.g. -4R
    worst   = loss_only_R.min()
    print(f"\n  Worst-day loss-only R exposure -> daily DD at each risk level:")
    print(f"  {'Risk%':>6} {'p5-day DD':>11} {'worst-day DD':>13} {'breaches -5%?':>14}")
    for rp in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        # DD from stacking that day's losers (approx, ignoring the winners that day)
        p5_dd    = (1 - np.prod([1 + R*rp/100 for R in
                    sorted([r for d in daily_batches for r in d if r<=0])[:0]])) if False else p5_loss*rp/100
        worst_dd = worst * rp / 100
        breach   = "YES" if abs(worst_dd) >= MAX_DAILY_LOSS else ("p5 YES" if abs(p5_loss*rp/100) >= MAX_DAILY_LOSS else "no")
        print(f"  {rp:>5.2f}% {p5_loss*rp/100*100:>10.2f}% {worst_dd*100:>12.2f}% {breach:>14}")
    return trades_per_day, day_R_sum, losers_per_day, loss_only_R


def main(risk_levels=None, runs=20_000):
    risk_levels = risk_levels or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    batches = load_daily_batches()

    print(f"\n{'='*100}")
    print(f"  FTMO BLOW-UP AUTOPSY  —  {runs:,} runs/level")
    print(f"{'='*100}")

    trades_pd, day_R, losers_pd, loss_R = analyse_daily_distribution(batches)

    print(f"\n  === CAUSE OF DEATH BY RISK LEVEL ===")
    print(f"  {'Risk%':>6} {'blow%':>7} {'daily%':>8} {'total%':>8} "
          f"{'medDeathDay':>12} {'fatalLosers':>12} {'fatalTrades':>12} {'fatalDayR':>10} {'gaveBack':>9}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*9}")

    rows = []
    for rp in risk_levels:
        m = analyse_risk(batches, rp, runs)
        rows.append(m)
        print(f"  {rp:>5.2f}% {100*m['blow_rate']:>6.1f}% "
              f"{100*m['pct_deaths_daily']:>7.1f}% {100*m['pct_deaths_total']:>7.1f}% "
              f"{m['med_day_of_death']:>12.0f} {m['med_fatal_losers']:>12.1f} "
              f"{m['med_fatal_trades']:>12.1f} {m['med_fatal_day_R']:>+10.2f} "
              f"{100*m['med_gaveback_total'] if not np.isnan(m['med_gaveback_total']) else 0:>8.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "ftmo_blowup_analysis.csv", index=False)
    print(f"\n  Data saved: {LOGS / 'ftmo_blowup_analysis.csv'}")

    _plot(df, trades_pd, day_R, loss_R, batches)
    print(f"{'='*100}\n")


def _plot(df, trades_pd, day_R, loss_R, batches):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = df["risk_pct"].values
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("FTMO Blow-up Autopsy — cause of death & the correlation-cluster killer",
                 color="#ffffff", fontsize=14, fontweight="bold")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#cccccc")
        ax.grid(color="#2a2a2a", linewidth=0.4)
        for s in ax.spines.values(): s.set_color("#333333")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_color("#cccccc")

    # Panel 1: stacked cause of death
    ax = axes[0, 0]
    ax.set_title("Cause of death: daily -5% vs total -10%", color="#cccccc")
    daily_share = 100*df["pct_deaths_daily"]*df["blow_rate"]
    total_share = 100*df["pct_deaths_total"]*df["blow_rate"]
    ax.bar(x, daily_share, width=0.12, color="#ff4444", label="Daily -5% breach")
    ax.bar(x, total_share, width=0.12, bottom=daily_share, color="#aa2222", label="Total -10% breach")
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("% of all challenges blown", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    # Panel 2: day of death
    ax = axes[0, 1]
    ax.set_title("How fast they die (median day of death)", color="#cccccc")
    ax.plot(x, df["med_day_of_death"], "-o", color="#ffaa00", linewidth=2)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("Median trading day of blow-up", color="#cccccc")

    # Panel 3: the killer day — losers that stacked
    ax = axes[1, 0]
    ax.set_title("The fatal day: losers stacked together", color="#cccccc")
    ax.plot(x, df["med_fatal_losers"], "-o", color="#ff6688", label="Median losers on fatal day", linewidth=2)
    ax.plot(x, df["med_fatal_trades"], "--o", color="#8888ff", label="Median trades on fatal day", linewidth=1.5)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("Trade count", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    # Panel 4: histogram of historical daily loss-only R (the raw ammunition)
    ax = axes[1, 1]
    ax.set_title("Historical daily loss-only R — the tail that kills", color="#cccccc")
    ax.hist(loss_R, bins=40, color="#ff4444", alpha=0.8, edgecolor="#331111")
    ax.axvline(np.percentile(loss_R, 5), color="#ffaa00", linewidth=1.5,
               linestyle="--", label=f"p5 = {np.percentile(loss_R,5):.1f}R")
    ax.axvline(loss_R.min(), color="#ffffff", linewidth=1.2,
               linestyle=":", label=f"worst = {loss_R.min():.1f}R")
    ax.set_xlabel("Sum of losing R on a single day", color="#cccccc")
    ax.set_ylabel("Number of days", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = LOGS / "ftmo_blowup_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=20_000)
    args = ap.parse_args()
    main(risk_levels=args.risk, runs=args.runs)
