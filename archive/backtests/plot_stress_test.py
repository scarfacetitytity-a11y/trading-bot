"""Visual equity curve and drawdown charts for the AiDEN stress test.

Generates per-risk-level charts:
  - Panel 1: Normalised equity curves, all instruments (log scale)
  - Panel 2: Portfolio combined equity + drawdown fill
  - Panel 3: Per-symbol max DD bar chart with FTMO limits
  - Panel 4: Trade count / WR scatter per instrument

Also runs H1 comparison for instruments with H1 data:
  XAUUSD, US30.cash, US100.cash

Usage:
    python -m backtests.plot_stress_test
    python -m backtests.plot_stress_test --risk 0.5 1.0 --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.run_stress_test import (
    run_single_risk, _portfolio_metrics, _daily_equity,
    FTMO_DAILY_LIMIT, FTMO_MAX_DD,
)
from backtests.tm_backtest import _ALL_TM_SYMBOLS
from backtests.run_multi_instrument import TRAIL_CONFIGS, OPTIMISED_PARAMS, BIDIRECTIONAL
from backtests.run_stress_test import _build_strategy, _load_symbol
from execution.trade_manager import TradeManager
from backtests.tm_backtest import simulate

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
LOGS      = Path(__file__).resolve().parent.parent / "logs"

H1_SYMBOLS = ["XAUUSD", "US30.cash", "US100.cash"]   # instruments with H1 data

PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff",
]


# ── H1 simulation wrapper ─────────────────────────────────────────────────────

def _run_h1(risk_pct: float, commission: float = 0.0001) -> list:
    from backtests.run_multi_instrument import H1_PARAMS, INSTRUMENTS
    from backtests.tm_backtest import _FX_PAIRS
    from strategies.aiden_index import AiDENIndexStrategy

    tm      = TradeManager()
    results = []

    for symbol in H1_SYMBOLS:
        p1 = PROCESSED / f"{symbol}_H1.csv"
        if not p1.exists():
            continue

        df1 = pd.read_csv(p1)
        df1["time"] = pd.to_datetime(df1["time"], utc=True, errors="coerce")
        df1 = df1.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        if len(df1) < 200:
            continue

        cfg       = INSTRUMENTS.get(symbol, {})
        tf_params = H1_PARAMS.copy()
        trail_cfg = TRAIL_CONFIGS.get(symbol, {})
        is_long_only = symbol not in BIDIRECTIONAL

        v2 = dict(
            long_only=is_long_only,
            rr_model3_bonus=0.5, rr_trend_bonus=0.5,
            rr_trend_threshold=0.003, rr_max=5.0,
            session_prime_start=13, session_prime_end=15,
            use_rsi=True, rsi_period=14,
            rsi_long_lo=25.0, rsi_long_hi=55.0,
            rsi_short_lo=45.0, rsi_short_hi=75.0,
            trail_to_be=trail_cfg.get("trail_to_be", False),
            trail_be_r=trail_cfg.get("trail_be_r", 1.0),
            trail_lock_r=trail_cfg.get("trail_lock_r", 2.0),
            t1_r=trail_cfg.get("t1_r", 0.0),
            t1_partial_pct=trail_cfg.get("t1_partial_pct", 0.5),
            time_stop_bars=0,
            use_prime_bonus=True, use_vol_spike=False,
            vol_spike_mult=1.5, require_ce=False, use_d1_bias=False,
        )

        if symbol in OPTIMISED_PARAMS:
            opt   = OPTIMISED_PARAMS[symbol].copy()
            bias  = opt.pop("h4_bias_method", "ema")
            stopb = opt.pop("atr_stop_buffer", 0.5)
            strat = AiDENIndexStrategy(
                min_score=opt.get("min_score", 4), min_fvg_atr=opt.get("min_fvg_atr", 0.10),
                rr_target=opt.get("rr_target", 2.5),
                session_start=opt.get("session_start", cfg.get("session_start", 7)),
                session_end=opt.get("session_end", cfg.get("session_end", 21)),
                h4_bias_method=bias, atr_stop_buffer=stopb,
                **{k: v for k, v in tf_params.items()
                   if k not in ("h4_bias_method", "atr_stop_buffer")},
                **v2,
            )
        else:
            bias  = tf_params.pop("h4_bias_method", "swing")
            stopb = tf_params.pop("atr_stop_buffer", 0.3)
            strat = AiDENIndexStrategy(
                min_score=4, min_fvg_atr=0.10, rr_target=2.5,
                session_start=cfg.get("session_start", 7),
                session_end=cfg.get("session_end", 21),
                h4_bias_method=bias, atr_stop_buffer=stopb,
                **tf_params, **v2,
            )

        try:
            res = simulate(
                df_m15=df1, df_m5=None, strat=strat, trade_manager=tm,
                symbol=symbol, initial_capital=1.0,
                risk_pct=risk_pct, commission=commission,
                daily_halt_pct=2.0,
            )
        except Exception as exc:
            print(f"  H1 {symbol} ERROR: {exc}")
            continue

        m  = res.metrics
        ts = df1["time"].reset_index(drop=True).iloc[:len(res.equity)]
        results.append({
            "symbol":  f"{symbol}_H1",
            "ret_pct": m["total_return_pct"],
            "dd_pct":  m["max_drawdown_pct"],
            "sharpe":  m["sharpe_ratio"],
            "trades":  m["total_trades"],
            "wr":      m.get("win_rate_pct", 0),
            "equity":  res.equity,
            "times":   ts,
        })
        print(f"  H1 {symbol:<14}  Ret={m['total_return_pct']:+.1f}%  "
              f"DD={m['max_drawdown_pct']:.1f}%  Sharpe={m['sharpe_ratio']:.2f}  "
              f"trades={m['total_trades']}")

    return results


# ── Chart builders ────────────────────────────────────────────────────────────

def _plot_risk_level(
    results:  list,
    port_met: dict,
    risk_pct: float,
    save_dir: Path,
    show:     bool = True,
) -> None:
    if not HAS_MPL:
        print("  matplotlib not installed — skipping charts")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"AiDEN Stress Test  —  {risk_pct}% risk/trade  |  "
                 f"FTMO limits: -{FTMO_MAX_DD}% max DD / -{FTMO_DAILY_LIMIT}% daily",
                 fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#cccccc")
        ax.spines[:].set_color("#333333")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#cccccc")

    # ── Panel 1: Normalised equity curves (log scale) ────────────────────────
    ax1 = axes[0, 0]
    ax1.set_title("Normalised Equity Curves (log)", color="#cccccc")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}x"))

    for i, r in enumerate(results):
        col  = PALETTE[i % len(PALETTE)]
        eq   = r.equity.values
        norm = eq / eq[0]
        ts   = pd.to_datetime(r.times.values, utc=True)
        ax1.plot(ts, norm, color=col, linewidth=0.8, label=r.symbol, alpha=0.9)

    ax1.axhline(1.0, color="#555555", linewidth=0.5, linestyle="--")
    ax1.legend(fontsize=7, loc="upper left", facecolor="#1a1a1a",
               labelcolor="#cccccc", ncol=2)
    ax1.grid(axis="y", color="#2a2a2a", linewidth=0.4)

    # ── Panel 2: Portfolio equity + drawdown fill ────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_title("Portfolio Combined Equity & Drawdown", color="#cccccc")

    if port_met and "port_equity" in port_met:
        peq  = port_met["port_equity"].values
        norm = peq / peq[0]

        # Reconstruct a date index for portfolio equity
        all_dates_sorted = sorted(set(
            d for r in results
            for d in _daily_equity(r.equity, r.times).index
        ))
        if len(all_dates_sorted) == len(norm):
            idx = pd.to_datetime(all_dates_sorted)
        else:
            idx = pd.RangeIndex(len(norm))

        ax2.plot(idx, norm, color="#00e5ff", linewidth=1.2, label="Portfolio")

        running_max = np.maximum.accumulate(norm)
        dd          = (norm - running_max) / running_max * 100
        ax2_dd = ax2.twinx()
        ax2_dd.fill_between(idx, dd, 0, alpha=0.3, color="#ff4444", label="DD%")
        ax2_dd.axhline(-FTMO_MAX_DD, color="#ff0000", linewidth=0.8, linestyle="--")
        ax2_dd.axhline(-FTMO_DAILY_LIMIT, color="#ff8800", linewidth=0.8, linestyle="--")
        ax2_dd.set_ylabel("Drawdown %", color="#ff4444", fontsize=8)
        ax2_dd.tick_params(colors="#cccccc")
        ax2_dd.set_facecolor("#1a1a1a")
        for label in ax2_dd.get_yticklabels():
            label.set_color("#cccccc")
        ax2.axhline(1.0, color="#555555", linewidth=0.5, linestyle="--")
        ax2.set_ylabel("Equity (norm)", color="#cccccc", fontsize=8)
        ax2.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a", labelcolor="#cccccc")
        ax2.grid(color="#2a2a2a", linewidth=0.4)

    # ── Panel 3: Per-symbol max DD bar chart ────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_title("Max Drawdown per Instrument", color="#cccccc")

    syms = [r.symbol for r in results]
    dds  = [r.max_dd_pct for r in results]
    cols = ["#ff4444" if d < -FTMO_MAX_DD else "#44ff88" for d in dds]
    bars = ax3.barh(syms, dds, color=cols, edgecolor="#333333", height=0.6)
    ax3.axvline(-FTMO_MAX_DD, color="#ff0000", linewidth=1.0, linestyle="--",
                label=f"FTMO {FTMO_MAX_DD}% limit")
    ax3.axvline(-FTMO_DAILY_LIMIT, color="#ff8800", linewidth=0.8, linestyle="--",
                label=f"FTMO {FTMO_DAILY_LIMIT}% daily")

    for bar, v in zip(bars, dds):
        ax3.text(v - 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", ha="right", fontsize=7, color="#cccccc")

    ax3.set_xlabel("Max DD %", color="#cccccc", fontsize=8)
    ax3.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="#cccccc")
    ax3.grid(axis="x", color="#2a2a2a", linewidth=0.4)
    ax3.tick_params(colors="#cccccc")
    ax3.invert_yaxis()
    for label in ax3.get_yticklabels():
        label.set_color("#cccccc")

    # ── Panel 4: Sharpe vs WR scatter ────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_title("Sharpe vs Win Rate", color="#cccccc")

    sharpes = [r.sharpe for r in results]
    wrs     = [r.wr for r in results]
    trades  = [max(1, r.trades) for r in results]

    sc = ax4.scatter(wrs, sharpes, s=[t * 0.6 for t in trades],
                     c=PALETTE[:len(results)], alpha=0.85, edgecolors="#333333",
                     linewidths=0.5)

    for r, w, s in zip(results, wrs, sharpes):
        ax4.annotate(r.symbol.replace(".cash", ""), (w, s), fontsize=6,
                     color="#cccccc", xytext=(3, 2), textcoords="offset points")

    ax4.axhline(1.0, color="#555555", linewidth=0.5, linestyle="--")
    ax4.axvline(50.0, color="#555555", linewidth=0.5, linestyle="--")
    ax4.set_xlabel("Win Rate %", color="#cccccc", fontsize=8)
    ax4.set_ylabel("Sharpe Ratio", color="#cccccc", fontsize=8)
    ax4.grid(color="#2a2a2a", linewidth=0.4)
    ax4.tick_params(colors="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fname = save_dir / f"stress_test_risk{str(risk_pct).replace('.', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved: {fname}")

    if show:
        plt.show()
    plt.close(fig)


def _plot_tf_comparison(
    m15_results: dict,
    h1_results:  list,
    save_dir:    Path,
    show:        bool = True,
) -> None:
    """Side-by-side M15 vs H1 equity for shared instruments."""
    if not HAS_MPL or not h1_results:
        return

    shared = [r["symbol"].replace("_H1", "") for r in h1_results]
    m15_map = {r.symbol: r for r in m15_results if r.symbol in shared}
    if not m15_map:
        return

    fig, axes = plt.subplots(len(h1_results), 2,
                              figsize=(16, 5 * len(h1_results)))
    if len(h1_results) == 1:
        axes = [axes]

    fig.suptitle("M15+M5 vs H1 Comparison  —  1.0% risk", fontsize=13,
                 fontweight="bold")
    fig.patch.set_facecolor("#0d0d0d")

    for i, h1r in enumerate(h1_results):
        sym    = h1r["symbol"].replace("_H1", "")
        m15r   = m15_map.get(sym)

        for j, (data, label, col) in enumerate([
            (m15r, "M15+M5", "#00e5ff"),
            (h1r,  "H1",     "#ff9900"),
        ]):
            ax = axes[i][j]
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc")
            ax.spines[:].set_color("#333333")

            if data is None:
                ax.set_title(f"{sym} {label} — no data", color="#666666")
                continue

            eq  = data["equity"].values if isinstance(data, dict) else data.equity.values
            ts  = data["times"].values  if isinstance(data, dict) else data.times.values

            norm = eq / eq[0]
            ax.plot(pd.to_datetime(ts, utc=True), norm, color=col, linewidth=0.9)

            m  = {}
            m["total_return_pct"] = (eq[-1] / eq[0] - 1) * 100
            running_max = np.maximum.accumulate(norm)
            dd = (norm - running_max) / running_max * 100
            m["max_dd_pct"] = float(dd.min())

            sharpe = data.get("sharpe", 0) if isinstance(data, dict) else data.sharpe
            trades = data.get("trades", 0) if isinstance(data, dict) else data.trades
            wr     = data.get("wr", 0)     if isinstance(data, dict) else data.wr

            title = (f"{sym} [{label}]  Ret={m['total_return_pct']:+.1f}%  "
                     f"DD={m['max_dd_pct']:.1f}%  Sharpe={sharpe:.2f}  "
                     f"Trades={trades}  WR={wr:.0f}%")
            ax.set_title(title, color="#cccccc", fontsize=9)

            ax_dd = ax.twinx()
            ax_dd.fill_between(pd.to_datetime(ts, utc=True), dd, 0,
                               alpha=0.2, color="#ff4444")
            ax_dd.axhline(-FTMO_MAX_DD, color="#ff0000", linewidth=0.6, linestyle="--")
            ax_dd.set_ylabel("DD%", color="#ff4444", fontsize=7)
            ax_dd.tick_params(colors="#cccccc")
            ax_dd.set_facecolor("#1a1a1a")
            for lbl in ax_dd.get_yticklabels():
                lbl.set_color("#cccccc")

            ax.set_ylabel("Equity (norm)", color=col, fontsize=7)
            ax.grid(color="#2a2a2a", linewidth=0.3)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_color("#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = save_dir / "tf_comparison_M15_vs_H1.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved: {fname}")
    if show:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    risk_levels: list[float] = [0.5, 1.0, 1.5],
    symbols:     Optional[list[str]] = None,
    show:        bool = True,
) -> None:
    if not HAS_MPL:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return

    LOGS.mkdir(exist_ok=True)

    if symbols is None:
        symbols = _ALL_TM_SYMBOLS

    print(f"\n{'='*70}")
    print(f"  AiDEN STRESS TEST — VISUAL CHARTS")
    print(f"  Risk levels : {risk_levels}")
    print(f"  Symbols     : {len(symbols)}")
    print(f"{'='*70}")

    all_m15_results = {}

    for risk in risk_levels:
        print(f"\n--- Risk {risk}% ---")
        results  = run_single_risk(risk_pct=risk, symbols=symbols)
        port_met = _portfolio_metrics(results, risk_pct=risk)
        all_m15_results[risk] = results

        _plot_risk_level(
            results=results, port_met=port_met,
            risk_pct=risk, save_dir=LOGS, show=show,
        )

    # H1 comparison at 1.0% risk
    print("\n--- H1 Comparison (1.0% risk) ---")
    h1_res = _run_h1(risk_pct=1.0)
    m15_at_1pct = all_m15_results.get(1.0, [])
    _plot_tf_comparison(
        m15_results=m15_at_1pct,
        h1_results=h1_res,
        save_dir=LOGS,
        show=show,
    )

    print(f"\nAll charts saved to {LOGS}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AiDEN Stress Test Charts")
    ap.add_argument("--risk",    type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--no-show", action="store_true", help="Save charts without displaying")
    ap.add_argument("symbols",   nargs="*", help="Symbols to include (default: all)")
    args = ap.parse_args()

    main(
        risk_levels=args.risk,
        symbols=args.symbols or None,
        show=not args.no_show,
    )
