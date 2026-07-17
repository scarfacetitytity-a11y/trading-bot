"""FTMO Phase 1 simulation with the LIVE ADAPTIVE risk stack (per-challenge).

`run_ftmo_sim.py` sizes every trade flat and bootstraps daily P&L buckets — that
cannot represent per-account gates (they'd have to reset each challenge). This
sim instead runs a fresh FTMO account per Monte-Carlo trial: it takes the REAL
multi-instrument trade flow of a random 30-day window and replays it, in true
entry/exit order, through the exact live stack:

    per-trade size = base_risk
        × score_mult          (0.75x @4, 1.0x @5, 1.5x @6+)
        × concentration_mult  (2x when <=1 open, 1.5x when <=3, else 1x)
        × psychology_mult      (0.5x weak rolling-WR; 0 when a gate halts)
        bounded by the portfolio aggregate-risk cap

    gates (reset every fresh challenge):
        consecutive-loss cooldown (3 -> 24h), rolling-WR halt (<=20%) / scale
        (<=35%), 2% daily soft-halt, 7% account-DD halt, 7% weekly-DD halt.

FTMO breaches (pass +10%, max-DD -10%, daily-DD -5%) are checked on the live
equity path, not daily buckets. Concentration/timing are preserved because each
trial is a contiguous real window, not resampled trades.

This answers "1% risk isn't our strategy": 1% is only the BASE — the stack varies
effective per-trade risk, and this measures the pass AND blow rate of that.

Usage
-----
    python -m backtests.run_ftmo_sim_adaptive --risk 1.0 --cap 4.0 --mc 5000 --compare
"""
from __future__ import annotations

import argparse
import heapq
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.engine import Backtest
from backtests.run_ftmo_sim import INSTRUMENTS, _build_strategy

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

_SIZE_CONFIGS = {4: 0.75, 5: 1.00}
_SIZE_DEFAULT = 1.5

TARGET = 0.10        # FTMO Phase 1 profit target (+10% of initial)
MAX_DD = 0.10        # max total loss (10% of initial)
DAILY_DD = 0.05      # max daily loss (5% of initial)
WINDOW_DAYS = 30     # calendar-day challenge window


def _score_mult(score: float) -> float:
    s = int(score) if score and not np.isnan(score) else 0
    if s <= 0:
        return 1.0
    return _SIZE_CONFIGS.get(s, _SIZE_DEFAULT)


def _concentration_mult(n_open: int) -> float:
    return 2.0 if n_open <= 1 else (1.5 if n_open <= 3 else 1.0)


# ── Trade loading with per-trade score ────────────────────────────────────────

def _load_trades_with_scores(score: int = 4) -> tuple[pd.DataFrame, float]:
    all_trades = []
    for symbol, cfg in INSTRUMENTS.items():
        path = PROCESSED / f"{symbol}_M15.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        strat = _build_strategy(symbol, score, cfg)
        bt    = Backtest(strat, initial_capital=10_000, commission=0.0001)
        r     = bt.run(df, symbol=symbol)

        trades = r.trades.copy()
        if trades.empty:
            continue
        # Map entry bar -> score. Use Timestamp keys (NOT .values / np.datetime64)
        # so the lookup dtype matches trades["entry_time"].
        score_map = dict(zip(df["time"], getattr(strat, "_scores")))
        trades["symbol"] = symbol
        trades["score"]  = trades["entry_time"].map(score_map).fillna(0)
        all_trades.append(trades)
        print(f"  {symbol:<16} {len(trades):>4} trades  "
              f"WR {(trades['pnl_pct']>0).mean()*100:.1f}%  "
              f"avg_score {trades.loc[trades['score']>0,'score'].mean():.2f}")

    if not all_trades:
        sys.exit("No trades found — run data pipeline first.")

    all_df = pd.concat(all_trades, ignore_index=True)
    all_df["entry_time"] = pd.to_datetime(all_df["entry_time"], utc=True)
    all_df["exit_time"]  = pd.to_datetime(all_df["exit_time"], utc=True)
    all_df = all_df.sort_values("entry_time").reset_index(drop=True)

    losses   = all_df[all_df["pnl_pct"] < 0]["pnl_pct"]
    avg_loss = float(losses.mean()) if len(losses) else -0.005
    return all_df, abs(avg_loss)


# ── Live pre-trade gate (fresh per challenge) ─────────────────────────────────

@dataclass
class _Gate:
    cum: float = 0.0                 # account return, fraction of initial
    peak: float = 0.0
    day: object = None
    day_start_cum: float = 0.0
    week: object = None
    week_start_cum: float = 0.0
    consec_losses: int = 0
    pause_until: object = None
    recent: list = field(default_factory=list)

    def _roll(self, ts):
        d = ts.normalize()
        if self.day is None or d != self.day:
            self.day, self.day_start_cum = d, self.cum
        wk = ts.isocalendar()[:2]
        if self.week is None or wk != self.week:
            self.week, self.week_start_cum = wk, self.cum

    def realize(self, ts, acct_ret, is_win):
        self._roll(ts)
        self.cum += acct_ret
        self.peak = max(self.peak, self.cum)
        self.recent.append(1 if is_win else 0)
        if len(self.recent) > 10:
            self.recent = self.recent[-10:]
        if is_win:
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            if self.consec_losses >= 3:
                self.pause_until = ts + pd.Timedelta(hours=24)

    def psych_mult(self, ts) -> float:
        self._roll(ts)
        if self.pause_until is not None and ts < self.pause_until:
            return 0.0
        if self.peak - self.cum >= 0.07:            # account DD halt
            return 0.0
        if self.day_start_cum - self.cum >= 0.02:   # 2% daily soft-halt
            return 0.0
        if self.week_start_cum - self.cum >= 0.07:  # weekly DD halt
            return 0.0
        if len(self.recent) >= 10:
            wr = sum(self.recent) / len(self.recent)
            if wr <= 0.20:
                return 0.0
            if wr <= 0.35:
                return 0.5
        return 1.0


# ── One fresh FTMO challenge ───────────────────────────────────────────────────

def _simulate_challenge(window: pd.DataFrame, avg_loss_abs: float,
                        base_risk_pct: float, cap_pct: float,
                        window_end, adaptive: bool = True,
                        daily_halt: float = 0.0) -> dict:
    """Replay one 30-day window as a fresh FTMO account. Returns result dict.

    daily_halt: if >0 (fraction, e.g. 0.03 = 3%), a HARD same-day loss halt that
    CLOSES all open exposure once the day's realized loss reaches it, then blocks
    entries for the rest of that day. Models a real daily kill that flattens
    positions — unlike the 2% soft-halt, which lets open trades run through it."""
    gate      = _Gate()
    base      = base_risk_pct / 100.0
    cap       = cap_pct / 100.0
    open_heap: list = []          # (exit_time, seq, acct_ret, is_win, risk)
    open_risk = 0.0
    eff: list = []
    seq = 0
    outcome = None
    day_i_hit = 0
    halted_day = None

    def _check_ftmo(ts) -> bool:
        """After an equity change: return True if the challenge resolved."""
        nonlocal outcome, day_i_hit
        if gate.day_start_cum - gate.cum >= DAILY_DD:
            outcome = "daily_dd"
        elif gate.cum <= -MAX_DD:
            outcome = "max_dd"
        elif gate.cum >= TARGET:
            outcome = "pass"
        if outcome:
            day_i_hit = (ts.normalize() - window_start).days + 1
            return True
        return False

    window_start = window_end - pd.Timedelta(days=WINDOW_DAYS)

    def _flush(until):
        nonlocal open_risk, halted_day
        while open_heap and open_heap[0][0] <= until:
            xt, _, acct_ret, is_win, risk = heapq.heappop(open_heap)
            gate.realize(xt, acct_ret, is_win)
            open_risk -= risk
            if _check_ftmo(xt):
                return True
            # Hard same-day loss halt: flatten all exposure, block rest of day.
            if daily_halt and (gate.day_start_cum - gate.cum) >= daily_halt:
                gate.cum = gate.day_start_cum - daily_halt
                open_heap.clear()
                open_risk = 0.0
                halted_day = gate.day
                return False
        return False

    for tr in window.itertuples(index=False):
        if _flush(tr.entry_time):
            return _result(outcome, day_i_hit, gate, eff)

        if halted_day is not None and tr.entry_time.normalize() == halted_day:
            continue   # day was flattened by the hard halt

        if adaptive:
            pm = gate.psych_mult(tr.entry_time)
            if pm == 0.0:
                continue
            mult = pm * _score_mult(tr.score) * _concentration_mult(len(open_heap))
            want = base * mult
            room = cap - open_risk
            if room <= 0.001:
                continue
            want = min(want, room)
        else:
            want = base   # flat baseline: same size every trade, no gates

        # fixed-fractional of current equity, expressed as fraction of initial
        acct_ret = tr.pnl_pct * (want / avg_loss_abs) * (1 + gate.cum)
        seq += 1
        heapq.heappush(open_heap, (tr.exit_time, seq, acct_ret, tr.pnl_pct > 0, want))
        open_risk += want
        eff.append(want * 100)

    # resolve remaining trades that exit within the 30-day window
    _flush(window_end)
    return _result(outcome or "timeout", day_i_hit, gate, eff)


def _result(outcome, day, gate, eff):
    return dict(result=outcome, day=day, final=gate.cum * 100,
                max_dd=(gate.peak - min(gate.cum, gate.peak)) * 100,
                eff=eff)


# ── Monte Carlo over real 30-day windows ──────────────────────────────────────

def _run_mc(trades, avg_loss_abs, base_risk, cap, mc, adaptive, rng):
    entries    = trades["entry_time"]
    first, last = entries.min(), entries.max()
    latest_start = last - pd.Timedelta(days=WINDOW_DAYS)
    span_days = max(1, (latest_start - first).days)

    results, all_eff = [], []
    ent = trades["entry_time"].values
    for _ in range(mc):
        start = first + pd.Timedelta(days=rng.randint(0, span_days))
        end   = start + pd.Timedelta(days=WINDOW_DAYS)
        lo = np.searchsorted(ent, np.datetime64(start))
        hi = np.searchsorted(ent, np.datetime64(end))
        if hi - lo < 3:
            continue
        r = _simulate_challenge(trades.iloc[lo:hi], avg_loss_abs, base_risk, cap,
                                end, adaptive=adaptive)
        results.append(r)
        all_eff.extend(r["eff"])

    n = len(results)
    passes = [r for r in results if r["result"] == "pass"]
    return dict(
        n=n,
        pass_pct=100 * len(passes) / n,
        blow_pct=100 * sum(r["result"] == "max_dd" for r in results) / n,
        daily_pct=100 * sum(r["result"] == "daily_dd" for r in results) / n,
        timeout_pct=100 * sum(r["result"] == "timeout" for r in results) / n,
        med_days=float(np.median([r["day"] for r in passes])) if passes else float("nan"),
        eff_mean=float(np.mean(all_eff)) if all_eff else 0.0,
        eff_p95=float(np.percentile(all_eff, 95)) if all_eff else 0.0,
        eff_max=float(np.max(all_eff)) if all_eff else 0.0,
    )


def run(score, base_risk, cap, mc, compare, seed=7):
    print(f"\n{'='*68}")
    print(f"  FTMO Phase 1 — ADAPTIVE live-stack sim  (base={base_risk}%  cap={cap}%)")
    print(f"{'='*68}\n")
    print("Building trade log (with per-trade scores)...")
    trades, avg_loss_abs = _load_trades_with_scores(score)
    print(f"\n  Total trades: {len(trades)}   avg_loss {-avg_loss_abs*100:.3f}%   "
          f"period {trades['entry_time'].min().date()}..{trades['entry_time'].max().date()}")

    a = _run_mc(trades, avg_loss_abs, base_risk, cap, mc, True, random.Random(seed))
    print(f"\n  {'='*64}")
    print(f"  ADAPTIVE   pass {a['pass_pct']:.1f}%   BLOW {a['blow_pct']:.2f}%   "
          f"daily-DD {a['daily_pct']:.2f}%   timeout {a['timeout_pct']:.1f}%")
    print(f"  {'='*64}")
    print(f"  windows simulated   : {a['n']}")
    print(f"  median days to pass : {a['med_days']:.0f}")
    print(f"  effective risk/trade: mean {a['eff_mean']:.2f}%  p95 {a['eff_p95']:.2f}%  "
          f"max {a['eff_max']:.2f}%   (base {base_risk}%)")

    if compare:
        f = _run_mc(trades, avg_loss_abs, base_risk, cap, mc, False, random.Random(seed))
        print(f"\n  {'—'*64}")
        print(f"  {'Metric':<22}{'FLAT '+str(base_risk)+'%':>14}{'ADAPTIVE':>14}")
        print(f"  {'—'*64}")
        for k, name in [("pass_pct", "Pass %"), ("blow_pct", "Blow % (max-DD)"),
                        ("daily_pct", "Daily-DD breach %"), ("timeout_pct", "Timeout %"),
                        ("med_days", "Median days")]:
            print(f"  {name:<22}{f[k]:>14.2f}{a[k]:>14.2f}")
    print(f"\n{'='*68}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score", type=int, default=4)
    p.add_argument("--risk",  type=float, default=1.0, help="BASE per-trade risk pct")
    p.add_argument("--cap",   type=float, default=4.0, help="Portfolio aggregate-risk cap pct")
    p.add_argument("--mc",    type=int,   default=5000)
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    run(a.score, a.risk, a.cap, a.mc, a.compare)


if __name__ == "__main__":
    main()
