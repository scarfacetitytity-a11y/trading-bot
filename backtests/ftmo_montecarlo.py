"""FTMO challenge Monte Carlo — pass rate, blow rate, days-to-pass vs risk level.

Built on the HONEST naive trade pool (logs/audit_naive_trades.csv): pure R,
stop-outs = -1.00R, no management inflation. Groups trades into trading-day
batches to preserve intraday clustering (a bad day can breach the 5% daily rule).

Each Monte Carlo challenge bootstraps trading days (with replacement) and applies
FTMO Phase 1 rules to a shared account:
  - Profit target : +10%   (Phase 1)
  - Max total loss: -10%   from initial balance (static floor)
  - Max daily loss: -5%    from that day's starting balance
  - Min trading days: 4    before a pass counts
  - Time cap       : 60 trading days -> "did not pass in time"

"Leverage" = risk % per trade. Higher risk = faster target BUT more blow-ups,
because N concurrent trades in a bad day compound into a daily-limit breach.

Also chains Phase 2 (+5% target, same DD rules) for a full-evaluation pass rate,
and estimates payout rate (reach funded + bank first 10%).

Usage:
    python -m backtests.ftmo_montecarlo
    python -m backtests.ftmo_montecarlo --risk 0.25 0.5 1.0 2.0 --runs 10000
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

# FTMO Phase 1 rules
PROFIT_TARGET_P1 = 0.10
PROFIT_TARGET_P2 = 0.05
MAX_TOTAL_LOSS   = 0.10   # static floor at 90% of initial
MAX_DAILY_LOSS   = 0.05
MIN_TRADING_DAYS = 4
TIME_CAP_DAYS    = 60

DEFAULT_RISK   = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_RUNS   = 10_000


def load_daily_batches(csv: Path = LOGS / "audit_naive_trades.csv") -> list[list[float]]:
    """Return list of trading days, each a list of R-outcomes (chronological within day)."""
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found — run: python -m backtests.audit_naive")
    df = pd.read_csv(csv)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["day"] = df["exit_time"].dt.date
    df = df.sort_values("exit_time")
    groups = defaultdict(list)
    for _, r in df.iterrows():
        groups[str(r["day"])].append(float(r["R"]))
    return [groups[d] for d in sorted(groups.keys())]


def simulate_phase(
    rng:          np.random.Generator,
    daily_batches: list[list[float]],
    risk_frac:    float,
    target:       float,
    start_equity: float = 1.0,
) -> tuple[str, int, float]:
    """Run one FTMO phase. Returns (outcome, days_used, end_equity).

    outcome in {"pass", "blown_total", "blown_daily", "timeout"}.
    Trades within a day applied sequentially; daily & total DD checked intrabar.
    """
    n_days   = len(daily_batches)
    equity   = start_equity
    floor    = start_equity * (1.0 - MAX_TOTAL_LOSS)     # static max-loss floor
    target_e = start_equity * (1.0 + target)

    for day in range(TIME_CAP_DAYS):
        batch = daily_batches[rng.integers(n_days)]
        day_start = equity
        daily_floor = day_start * (1.0 - MAX_DAILY_LOSS)

        for R in batch:
            equity *= (1.0 + R * risk_frac)
            if equity <= floor:
                return "blown_total", day + 1, equity
            if equity <= daily_floor:
                return "blown_daily", day + 1, equity

        if equity >= target_e and (day + 1) >= MIN_TRADING_DAYS:
            return "pass", day + 1, equity

    return "timeout", TIME_CAP_DAYS, equity


def run_risk_level(
    daily_batches: list[list[float]],
    risk_pct:     float,
    runs:         int,
    seed:         int = 42,
) -> dict:
    rng       = np.random.default_rng(seed)
    risk_frac = risk_pct / 100.0

    p1_pass = p1_blown = p1_timeout = 0
    full_pass = 0            # passed P1 AND P2
    payout    = 0            # passed eval + banked first +10% funded
    days_to_pass_p1: list[int] = []
    days_full:       list[int] = []

    for _ in range(runs):
        out1, d1, eq1 = simulate_phase(rng, daily_batches, risk_frac, PROFIT_TARGET_P1)
        if out1 == "pass":
            p1_pass += 1
            days_to_pass_p1.append(d1)
            # Phase 2 — fresh equity base, +5% target
            out2, d2, eq2 = simulate_phase(rng, daily_batches, risk_frac, PROFIT_TARGET_P2)
            if out2 == "pass":
                full_pass += 1
                # Funded: bank first +10% without breaching -> payout
                out3, d3, eq3 = simulate_phase(rng, daily_batches, risk_frac, 0.10)
                if out3 == "pass":
                    payout += 1
                    days_full.append(d1 + d2 + d3)
        elif out1 in ("blown_total", "blown_daily"):
            p1_blown += 1
        else:
            p1_timeout += 1

    return {
        "risk_pct":        risk_pct,
        "p1_pass_rate":    p1_pass / runs,
        "p1_blow_rate":    p1_blown / runs,
        "p1_timeout_rate": p1_timeout / runs,
        "full_pass_rate":  full_pass / runs,
        "payout_rate":     payout / runs,
        "median_days_p1":  float(np.median(days_to_pass_p1)) if days_to_pass_p1 else float("nan"),
        "p10_days_p1":     float(np.percentile(days_to_pass_p1, 10)) if days_to_pass_p1 else float("nan"),
        "median_days_full": float(np.median(days_full)) if days_full else float("nan"),
    }


def main(risk_levels=None, runs=DEFAULT_RUNS, save_chart=True) -> None:
    risk_levels = risk_levels or DEFAULT_RISK
    daily_batches = load_daily_batches()

    # pool summary
    all_R = np.array([r for day in daily_batches for r in day])
    print(f"\n{'='*92}")
    print(f"  FTMO CHALLENGE MONTE CARLO  —  {runs:,} runs/level  —  HONEST naive pool")
    print(f"  Pool: {len(all_R)} trades, {len(daily_batches)} days, "
          f"expectancy {all_R.mean():+.3f}R, WR {100*(all_R>0).mean():.1f}%, "
          f"median {np.median([len(d) for d in daily_batches]):.0f} trades/day")
    print(f"  Rules: +10% target | -10% max loss | -5% daily | min 4 days | {TIME_CAP_DAYS}-day cap")
    print(f"{'='*92}")
    print(f"  {'Risk%':>6} {'P1 pass':>8} {'P1 blow':>8} {'timeout':>8} "
          f"{'full pass':>10} {'payout':>8} {'med days':>9} {'fast(p10)':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*9} {'-'*10}")

    rows = []
    for rp in risk_levels:
        m = run_risk_level(daily_batches, rp, runs)
        rows.append(m)
        print(f"  {rp:>5.2f}% {100*m['p1_pass_rate']:>7.1f}% {100*m['p1_blow_rate']:>7.1f}% "
              f"{100*m['p1_timeout_rate']:>7.1f}% {100*m['full_pass_rate']:>9.1f}% "
              f"{100*m['payout_rate']:>7.1f}% {m['median_days_p1']:>9.0f} {m['p10_days_p1']:>10.0f}")

    df = pd.DataFrame(rows)
    out_csv = LOGS / "ftmo_montecarlo.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Data saved: {out_csv}")

    # ── Find sweet spot ──
    best_full = df.loc[df["full_pass_rate"].idxmax()]
    best_ev   = df.copy()
    best_ev["ev_proxy"] = best_ev["payout_rate"]   # payout rate is the money metric
    best_pay  = best_ev.loc[best_ev["ev_proxy"].idxmax()]
    print(f"\n  Highest full-eval pass rate : {best_full['risk_pct']:.2f}% risk "
          f"({100*best_full['full_pass_rate']:.1f}%)")
    print(f"  Highest payout rate         : {best_pay['risk_pct']:.2f}% risk "
          f"({100*best_pay['payout_rate']:.1f}%)")

    if save_chart:
        _plot(df, all_R, len(daily_batches))
    print(f"{'='*92}\n")


def _plot(df: pd.DataFrame, all_R: np.ndarray, n_days: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing — skipping chart")
        return

    x = df["risk_pct"].values
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"FTMO Challenge Monte Carlo — risk-per-trade vs outcome  "
        f"(honest edge: {all_R.mean():+.2f}R/trade, {100*(all_R>0).mean():.0f}% WR)",
        color="#ffffff", fontsize=14, fontweight="bold")

    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#cccccc")
        ax.grid(color="#2a2a2a", linewidth=0.4)
        for s in ax.spines.values():
            s.set_color("#333333")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color("#cccccc")

    # Panel 1: pass vs blow vs timeout
    ax = axes[0, 0]
    ax.set_title("Phase 1 outcome vs risk", color="#cccccc")
    ax.plot(x, 100*df["p1_pass_rate"], "-o", color="#44ff88", label="Pass", linewidth=2)
    ax.plot(x, 100*df["p1_blow_rate"], "-o", color="#ff4444", label="BLOWN", linewidth=2)
    ax.plot(x, 100*df["p1_timeout_rate"], "-o", color="#ffaa00", label="Timeout", linewidth=1.5)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("% of challenges", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    # Panel 2: full-eval pass + payout
    ax = axes[0, 1]
    ax.set_title("Full evaluation pass & payout rate", color="#cccccc")
    ax.plot(x, 100*df["full_pass_rate"], "-o", color="#00e5ff", label="Full pass (P1+P2)", linewidth=2)
    ax.plot(x, 100*df["payout_rate"], "-o", color="#bb88ff", label="Reach payout", linewidth=2)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("% of challenges", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    # Panel 3: days to pass
    ax = axes[1, 0]
    ax.set_title("Speed — trading days to pass Phase 1", color="#cccccc")
    ax.plot(x, df["median_days_p1"], "-o", color="#44ff88", label="Median", linewidth=2)
    ax.plot(x, df["p10_days_p1"], "--o", color="#88ffcc", label="Fast (p10)", linewidth=1.5)
    ax.set_xlabel("Risk per trade %", color="#cccccc")
    ax.set_ylabel("Trading days", color="#cccccc")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")

    # Panel 4: the tradeoff — pass rate vs blow rate scatter, sized by speed
    ax = axes[1, 1]
    ax.set_title("The tradeoff: reward vs ruin", color="#cccccc")
    sizes = 400 / np.clip(df["median_days_p1"].values, 1, None)   # faster = bigger
    sc = ax.scatter(100*df["p1_blow_rate"], 100*df["full_pass_rate"],
                    s=200, c=x, cmap="plasma", edgecolors="#ffffff", linewidths=0.6, zorder=3)
    for xi, bi, pi in zip(x, 100*df["p1_blow_rate"], 100*df["full_pass_rate"]):
        ax.annotate(f"{xi:.2f}%", (bi, pi), color="#ffffff", fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Blow-up rate % (ruin)", color="#cccccc")
    ax.set_ylabel("Full pass rate % (reward)", color="#cccccc")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Risk %", color="#cccccc")
    cb.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = LOGS / "ftmo_montecarlo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, nargs="+", default=DEFAULT_RISK)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = ap.parse_args()
    main(risk_levels=args.risk, runs=args.runs)
