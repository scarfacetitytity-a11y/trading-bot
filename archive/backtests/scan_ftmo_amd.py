"""ICT AMD Breaker — multi-pair, multi-timeframe backtest + FTMO pass rate by risk level.

Best config: swing_lookbacks=[288], zone_atr=2.0, max_wait=5

Shows Phase 1 pass rate across a sweep of risk multipliers so you can compare
how the odds change between a $10k account (where traders typically push 2-3%
risk per trade) and a $100k account (where 0.5-1% is more common).

FTMO percentage rules are identical for all account sizes — the difference
between $10k and $100k is purely how aggressively you size each trade.

FTMO Phase 1: 10% target | 5% daily loss | 10% total loss | 30 days | 4 min trade days
FTMO Phase 2:  5% target | 5% daily loss | 10% total loss | 60 days | 4 min trade days

Usage (project root, venv active):
    python -m backtests.scan_ftmo_amd
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd

from backtests.engine import Backtest
from strategies.ict_amd import ICTAMDBreakerStrategy

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

PAIRS      = ["XAUUSD", "GBPUSD", "US100.cash"]
TIMEFRAMES = ["M5", "M15", "H1", "D1"]

STRATEGY_PARAMS = dict(swing_lookbacks=[288], zone_atr=2.0, max_wait=5)

# Risk levels to sweep.  Label shows which account profile each maps to.
RISK_LEVELS = [
    (0.5,  "$100k conservative"),
    (1.0,  "$100k standard / $10k conservative"),
    (1.5,  "$10k moderate"),
    (2.0,  "$10k standard"),
    (3.0,  "$10k aggressive"),
]

# ── FTMO rules ────────────────────────────────────────────────────────────────
P1_TARGET     = 0.10
P1_DAILY_LOSS = 0.05
P1_TOTAL_LOSS = 0.10
P1_MIN_DAYS   = 4
P1_WINDOW     = 30

P2_TARGET     = 0.05
P2_DAILY_LOSS = 0.05
P2_TOTAL_LOSS = 0.10
P2_MIN_DAYS   = 4
P2_WINDOW     = 60

STEP_DAYS = 7
# ─────────────────────────────────────────────────────────────────────────────


def _roll_windows(
    equity: pd.Series,
    trades: pd.DataFrame,
    profit_target: float,
    max_daily_loss: float,
    max_total_loss: float,
    min_trade_days: int,
    window_days: int,
    risk_mult: float,
) -> dict:
    all_dates = equity.index.normalize().unique()
    results   = []
    start     = all_dates[0]

    while True:
        candidates = all_dates[all_dates >= start]
        if len(candidates) == 0:
            break
        w_start = candidates[0]
        if w_start + timedelta(days=window_days) > all_dates[-1]:
            break

        end    = w_start + timedelta(days=window_days)
        mask   = (equity.index >= w_start) & (equity.index < end)
        eq_win = equity[mask]

        if len(eq_win) > 0:
            if risk_mult != 1.0:
                bar_rets = eq_win.pct_change().fillna(0)
                eq_norm  = (1 + bar_rets * risk_mult).cumprod()
            else:
                eq_norm = eq_win / eq_win.iloc[0]

            bust_reason = None
            target_hit  = False
            peak_today  = 1.0
            current_day = None
            final_val   = eq_norm.iloc[-1]

            for ts, val in eq_norm.items():
                day = ts.date()
                if day != current_day:
                    peak_today  = val
                    current_day = day
                else:
                    peak_today = max(peak_today, val)

                if (val - peak_today) / peak_today <= -max_daily_loss:
                    bust_reason = "daily"
                    final_val   = val
                    break
                if (val - 1.0) <= -max_total_loss:
                    bust_reason = "total"
                    final_val   = val
                    break
                if val >= 1.0 + profit_target:
                    target_hit = True
                    final_val  = val
                    break

            trade_days = 0
            if len(trades) > 0:
                t_mask     = (pd.to_datetime(trades["exit_time"]) >= w_start) & \
                             (pd.to_datetime(trades["exit_time"]) < end)
                trade_days = pd.to_datetime(trades[t_mask]["exit_time"]).dt.date.nunique()

            passed = target_hit and bust_reason is None and trade_days >= min_trade_days
            results.append({
                "passed": passed,
                "busted": bust_reason is not None,
                "return": final_val - 1.0,
            })

        nxt = all_dates[all_dates >= start + timedelta(days=STEP_DAYS)]
        if len(nxt) == 0:
            break
        start = nxt[0]

    if not results:
        return {"windows": 0, "pass_rate": 0.0, "bust_rate": 0.0, "avg_return": 0.0}

    df = pd.DataFrame(results)
    n  = len(df)
    return {
        "windows":    n,
        "pass_rate":  df["passed"].sum() / n * 100,
        "bust_rate":  df["busted"].sum() / n * 100,
        "avg_return": df["return"].mean() * 100,
    }


def run() -> None:
    strat    = ICTAMDBreakerStrategy(**STRATEGY_PARAMS)
    all_rows = []

    risk_mults  = [r for r, _ in RISK_LEVELS]
    risk_labels = [lbl for _, lbl in RISK_LEVELS]

    # ── header ────────────────────────────────────────────────────────────────
    col_w = 8
    risk_header = "  ".join(f"×{r:<3}" for r in risk_mults)

    print("\n" + "=" * 100)
    print("  ICT AMD Breaker | swing=288  zone_atr=2.0  max_wait=5")
    print("  FTMO Phase 1 pass rate by risk multiplier (×1.0 = backtest default ~1% risk/trade)")
    print()
    print(f"  {'':30}  {'< $100k typical >':^20}  {'< $10k typical >':^25}")
    risk_line = "  ".join(f"×{r:<3}" for r in risk_mults)
    print(f"  {'Pair + Timeframe':<16}  {'Trades':>6}  {'WR%':>5}  {'Ret%':>6}  {'Sharpe':>6}  "
          f"{'MaxDD':>5}  {risk_line}   {'Bust@×2':>7}  {'Windows':>7}")
    print("  " + "-" * 96)

    for pair in PAIRS:
        printed_pair = False
        for tf in TIMEFRAMES:
            path = PROCESSED_DIR / f"{pair}_{tf}.csv"
            if not path.exists():
                continue

            label = f"{pair} {tf}"
            try:
                raw = pd.read_csv(path)
                bt  = Backtest(strat, initial_capital=10_000, commission=0.0001)
                r   = bt.run(raw, symbol=pair)
                m   = r.metrics

                if m["total_trades"] == 0:
                    print(f"  {label:<16}  — no trades")
                    continue

                eq       = r.equity.copy()
                eq.index = pd.to_datetime(raw["time"])

                # Phase 1 pass rates at each risk level
                p1_rates = []
                bust_at_2 = 0.0
                windows   = 0
                for rm in risk_mults:
                    sim = _roll_windows(
                        eq, r.trades,
                        P1_TARGET, P1_DAILY_LOSS, P1_TOTAL_LOSS,
                        P1_MIN_DAYS, P1_WINDOW, rm,
                    )
                    p1_rates.append(sim["pass_rate"])
                    if rm == 2.0:
                        bust_at_2 = sim["bust_rate"]
                        windows   = sim["windows"]

                # Phase 2 at ×1.0 baseline
                p2_base = _roll_windows(
                    eq, r.trades,
                    P2_TARGET, P2_DAILY_LOSS, P2_TOTAL_LOSS,
                    P2_MIN_DAYS, P2_WINDOW, 1.0,
                )

                rates_str = "  ".join(f"{v:>5.1f}%" for v in p1_rates)

                print(
                    f"  {label:<16}  {m['total_trades']:>6}  "
                    f"{m['win_rate_pct']:>5.1f}%  "
                    f"{m['total_return_pct']:>+5.2f}%  "
                    f"{m['sharpe_ratio']:>6.3f}  "
                    f"{m['max_drawdown_pct']:>4.1f}%  "
                    f"{rates_str}   "
                    f"{bust_at_2:>6.1f}%  "
                    f"{windows:>7}"
                )

                for rm, p1r in zip(risk_mults, p1_rates):
                    all_rows.append({
                        "pair": pair, "timeframe": tf,
                        "trades": m["total_trades"],
                        "win_rate": round(m["win_rate_pct"], 1),
                        "total_return": round(m["total_return_pct"], 2),
                        "sharpe": round(m["sharpe_ratio"], 3),
                        "max_dd": round(m["max_drawdown_pct"], 2),
                        "risk_mult": rm,
                        "p1_pass_rate": round(p1r, 1),
                        "p2_pass_rate_at_1x": round(p2_base["pass_rate"], 1),
                        "bust_rate_at_risk": round(bust_at_2, 1),
                        "windows": windows,
                    })

            except Exception as exc:
                print(f"  {label:<16}  ERROR: {exc}")

        print()  # blank line between pairs

    if not all_rows:
        print("No results.")
        return

    # ── risk-level legend ─────────────────────────────────────────────────────
    print("  Risk level guide:")
    for rm, lbl in RISK_LEVELS:
        print(f"    ×{rm:<4}  {lbl}")

    # ── Phase 2 summary ───────────────────────────────────────────────────────
    print()
    print("  FTMO Phase 2 pass rate at ×1.0 risk (5% target, 60 days):")
    seen = set()
    for row in all_rows:
        key = (row["pair"], row["timeframe"])
        if key not in seen and row["risk_mult"] == 1.0:
            seen.add(key)
            print(f"    {row['pair']} {row['timeframe']:<5}  "
                  f"P2={row['p2_pass_rate_at_1x']:>5.1f}%")

    # ── best combo by P1 pass at ×2.0 (typical $10k standard) ───────────────
    df = pd.DataFrame(all_rows)
    top = df[df["risk_mult"] == 2.0].sort_values("p1_pass_rate", ascending=False)
    print()
    print("  TOP 5 — Phase 1 pass rate at ×2.0 risk ($10k standard):")
    for _, row in top.head(5).iterrows():
        print(f"    {row['pair']} {row['timeframe']:<5}  "
              f"P1={row['p1_pass_rate']:>5.1f}%  "
              f"WR={row['win_rate']:>5.1f}%  "
              f"Sharpe={row['sharpe']:>6.3f}  "
              f"trades={row['trades']:>4}  "
              f"bust={row['bust_rate_at_risk']:>5.1f}%")

    top100 = df[df["risk_mult"] == 1.0].sort_values("p1_pass_rate", ascending=False)
    print()
    print("  TOP 5 — Phase 1 pass rate at ×1.0 risk ($100k standard):")
    for _, row in top100.head(5).iterrows():
        print(f"    {row['pair']} {row['timeframe']:<5}  "
              f"P1={row['p1_pass_rate']:>5.1f}%  "
              f"WR={row['win_rate']:>5.1f}%  "
              f"Sharpe={row['sharpe']:>6.3f}  "
              f"trades={row['trades']:>4}")

    out = Path("logs") / "scan_ftmo_amd_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Full results saved to {out}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run()
