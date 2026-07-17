"""Bar-by-bar FTMO Phase 1 simulator with the live adaptive risk stack.

The trade-level sims replay finished trades — they cannot see a position's live
P&L mid-trade, so they can't model reallocation, an active daily-loss guard, or
FTMO breaches on the *floating* equity path. This one steps M15 bar by bar:

  - real strategy signals / stops / scores per instrument
  - intrabar SL/TP fills (pessimistic: stop checked before target)
  - mark-to-market every bar; FTMO daily/total/target checked on equity INCLUDING
    open floating P&L (as the broker measures it)
  - sizing + reallocation + active daily-loss guard driven by the REAL
    PortfolioAllocator; psychology/soft-halt gates via the shared _Gate
  - fresh account per Monte-Carlo 30-day window

Compares FLAT vs ADAPTIVE on identical windows so we can see whether the adaptive
stack lets us take MORE well-managed trades (higher pass) rather than fewer.

Usage
-----
    python -m backtests.run_ftmo_barsim --risk 1.0 --cap 4.0 --mc 400 --compare
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.run_ftmo_sim import INSTRUMENTS, OPTIMISED, _build_strategy
from backtests.run_ftmo_sim_adaptive import _Gate, _score_mult, _concentration_mult
from execution.portfolio_manager import PortfolioAllocator, OpenPos

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGET, MAX_DD, DAILY_DD, WINDOW_DAYS = 0.10, 0.10, 0.05, 30

# Trailing defaults (mirror live TRAIL_CONFIGS intent): BE at +1R, lock +1R at +2R,
# T1 partial 50% at +1R.
BE_R, LOCK_R, T1_R, T1_PCT = 1.0, 2.0, 1.0, 0.5


@dataclass
class Pos:
    symbol: str
    dir: int
    entry: float
    sl: float
    tp: float
    risk_dist: float
    risk_pct: float          # current risk as % of initial (shrinks on T1/trims)
    score: int
    be_done: bool = False
    t1_done: bool = False

    def floating_r(self, price: float) -> float:
        return (price - self.entry) / self.risk_dist * self.dir


# ── Per-instrument precompute ─────────────────────────────────────────────────

def _load_symbol(symbol: str, cfg: dict, score: int):
    path = PROCESSED / f"{symbol}_M15.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    strat = _build_strategy(symbol, score, cfg)
    sig = strat.generate_signals(df).to_numpy()
    stops = getattr(strat, "_stops").to_numpy()
    scores = getattr(strat, "_scores").to_numpy()
    rr = float(OPTIMISED.get(symbol, {}).get("rr_target", 2.5))
    return dict(
        symbol=symbol, time=df["time"].to_numpy(),
        open=df["open"].to_numpy(), high=df["high"].to_numpy(),
        low=df["low"].to_numpy(), close=df["close"].to_numpy(),
        sig=sig, stops=stops, scores=scores, rr=rr,
    )


def _build_events(data: dict):
    """Flat, time-sorted list of (timestamp, symbol, bar_index)."""
    ev = []
    for sym, d in data.items():
        for i in range(len(d["time"])):
            ev.append((d["time"][i], sym, i))
    ev.sort(key=lambda e: e[0])
    return ev


# ── One fresh challenge, stepped bar by bar ───────────────────────────────────

def _simulate(events, data, start, end, base_risk, cap, adaptive, alloc, start_idx=0,
              guard_buffer=1.0):
    gate = _Gate()
    base = base_risk / 100.0
    open_pos: dict = {}          # symbol -> Pos
    realized = 0.0               # account return fraction (closed trades)
    day = None
    day_start_eq = 0.0
    outcome = None
    day_hit = 0
    taken = 0

    def total_equity(ts_idx_by_sym):
        flo = 0.0
        for sym, p in open_pos.items():
            px = data[sym]["close"][ts_idx_by_sym[sym]]
            flo += p.floating_r(px) * p.risk_pct / 100.0
        return realized + flo

    # track latest bar index seen per symbol (for MTM of other symbols)
    last_idx: dict = {}

    def _resolve(ts):
        nonlocal outcome, day_hit
        eq = realized + sum(
            p.floating_r(data[s]["close"][last_idx[s]]) * p.risk_pct / 100.0
            for s, p in open_pos.items() if s in last_idx
        )
        if day_start_eq - eq >= DAILY_DD:
            outcome = "daily_dd"
        elif eq <= -MAX_DD:
            outcome = "max_dd"
        elif eq >= TARGET:
            outcome = "pass"
        if outcome:
            day_hit = (ts.normalize() - start.normalize()).days + 1
            return True
        return False

    def _close(p: Pos, exit_price: float, ts):
        nonlocal realized
        r = p.floating_r(exit_price)
        realized += r * p.risk_pct / 100.0
        gate.realize(ts, r * p.risk_pct / 100.0, r > 0)

    for k in range(start_idx, len(events)):
        ts, sym, i = events[k]
        if ts >= end:
            break
        d = data[sym]
        last_idx[sym] = i

        # New calendar day → re-anchor daily baseline (server day ~ UTC here)
        dd = ts.normalize()
        if day is None or dd != day:
            day = dd
            day_start_eq = total_equity(last_idx)

        # ── 1. intrabar SL/TP fill for this symbol's open position ────────────
        p = open_pos.get(sym)
        if p is not None:
            hi, lo = d["high"][i], d["low"][i]
            hit_sl = (p.dir == 1 and lo <= p.sl) or (p.dir == -1 and hi >= p.sl)
            hit_tp = (p.dir == 1 and hi >= p.tp) or (p.dir == -1 and lo <= p.tp)
            if hit_sl:                       # pessimistic: stop first
                _close(p, p.sl, ts); del open_pos[sym]; p = None
            elif hit_tp:
                _close(p, p.tp, ts); del open_pos[sym]; p = None
            if _resolve(ts):
                return dict(result=outcome, day=day_hit, taken=taken)

        # ── 2. management: BE / lock / T1 on the surviving position ───────────
        p = open_pos.get(sym)
        if p is not None:
            px = d["close"][i]
            r = p.floating_r(px)
            if not p.t1_done and r >= T1_R:
                # bank T1 partial, shrink risk, move to BE
                realized += r * (p.risk_pct * T1_PCT) / 100.0
                p.risk_pct *= (1 - T1_PCT)
                p.t1_done = True
                p.sl = p.entry
            if r >= LOCK_R:
                p.sl = p.entry + p.risk_dist * p.dir      # lock +1R
            elif r >= BE_R and not p.be_done:
                p.sl = p.entry; p.be_done = True

        # ── 3. active daily-loss guard (behaviour 3) ──────────────────────────
        if adaptive and open_pos:
            eq = total_equity(last_idx)
            day_loss = max(0.0, (day_start_eq - eq) * 100)
            book = [OpenPos(s, q.score, q.risk_pct, q.dir, q.floating_r(data[s]["close"][last_idx[s]]))
                    for s, q in open_pos.items()]
            for tr in alloc.daily_guard(day_loss, book, limit_pct=DAILY_DD*100, buffer_pct=guard_buffer):
                q = open_pos.get(tr.symbol)
                if q is None:
                    continue
                frac = min(1.0, tr.reduce_pct / q.risk_pct) if q.risk_pct > 0 else 0
                if frac > 0:
                    realized += q.floating_r(data[tr.symbol]["close"][last_idx[tr.symbol]]) * (q.risk_pct*frac)/100.0
                    q.risk_pct *= (1 - frac)

        # ── 4. entry: signal flips to nonzero and symbol is flat ──────────────
        sig_i = d["sig"][i]
        prev  = d["sig"][i-1] if i > 0 else 0
        if sym in open_pos:
            # exit on flip to flat/opposite
            if sig_i != open_pos[sym].dir and sig_i != 0:
                _close(open_pos[sym], d["close"][i], ts); del open_pos[sym]
            elif sig_i == 0:
                _close(open_pos[sym], d["close"][i], ts); del open_pos[sym]
        if sym not in open_pos and sig_i != 0 and sig_i != prev:
            stop = d["stops"][i]
            entry = d["close"][i]
            if stop and not np.isnan(stop) and stop > 0:
                risk_dist = abs(entry - stop)
                if risk_dist > 0:
                    score = int(d["scores"][i]) if d["scores"][i] else 5
                    n_open = len(open_pos)
                    mult = _score_mult(score) * _concentration_mult(n_open)
                    intended = base_risk * mult
                    if adaptive:
                        pm = gate.psych_mult(ts)
                        if pm > 0:
                            book = [OpenPos(s, q.score, q.risk_pct, q.dir,
                                            q.floating_r(data[s]["close"][last_idx[s]]))
                                    for s, q in open_pos.items()]
                            a = alloc.allocate(score, intended * pm, book)
                            granted = a.granted_pct
                            for tr in a.trims:
                                q = open_pos.get(tr.symbol)
                                if q and q.risk_pct > 0:
                                    frac = min(1.0, tr.reduce_pct / q.risk_pct)
                                    realized += q.floating_r(data[tr.symbol]["close"][last_idx[tr.symbol]]) * (q.risk_pct*frac)/100.0
                                    q.risk_pct *= (1 - frac)
                        else:
                            granted = 0.0
                    else:
                        # flat: bound to remaining cap
                        open_risk = sum(q.risk_pct for q in open_pos.values())
                        granted = max(0.0, min(intended, cap - open_risk))
                    if granted > 0:
                        tp = entry + d["rr"] * risk_dist * sig_i
                        open_pos[sym] = Pos(sym, int(sig_i), entry, stop, tp,
                                            risk_dist, granted, score)
                        taken += 1
        if _resolve(ts):
            return dict(result=outcome, day=day_hit, taken=taken)

    return dict(result=outcome or "timeout", day=day_hit, taken=taken)


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def _run(events, data, base_risk, cap, adaptive, mc, rng, guard_buffer=1.0,
         score_edge=1, min_trade=0.25):
    import bisect
    times = [e[0] for e in events]
    first, last = times[0], times[-1]
    latest = last - pd.Timedelta(days=WINDOW_DAYS)
    span = max(1, (latest - first).days)
    alloc = PortfolioAllocator(daily_budget_pct=cap, score_edge=score_edge,
                               min_trade_pct=min_trade)
    res, takes = [], []
    for _ in range(mc):
        start = first + pd.Timedelta(days=rng.randint(0, span))
        start_idx = bisect.bisect_left(times, start)      # skip the O(all events) scan
        r = _simulate(events, data, start, start + pd.Timedelta(days=WINDOW_DAYS),
                      base_risk, cap, adaptive, alloc, start_idx=start_idx, guard_buffer=guard_buffer)
        res.append(r); takes.append(r["taken"])
    n = len(res)
    passes = [r for r in res if r["result"] == "pass"]
    return dict(
        n=n,
        pass_pct=100*len(passes)/n,
        blow_pct=100*sum(r["result"]=="max_dd" for r in res)/n,
        daily_pct=100*sum(r["result"]=="daily_dd" for r in res)/n,
        timeout_pct=100*sum(r["result"]=="timeout" for r in res)/n,
        med_days=float(np.median([r["day"] for r in passes])) if passes else float("nan"),
        avg_trades=float(np.mean(takes)) if takes else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=int, default=4)
    ap.add_argument("--risk",  type=float, default=1.0)
    ap.add_argument("--cap",   type=float, default=4.0)
    ap.add_argument("--mc",    type=int,   default=400)
    ap.add_argument("--buffer", type=float, default=1.0, help="daily-guard buffer pct")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    print(f"\n{'='*70}\n  FTMO Phase 1 — BAR-BY-BAR sim  (risk={a.risk}%  cap={a.cap}%)\n{'='*70}\n")
    print("Loading + signalling instruments...")
    data = {}
    for sym, cfg in INSTRUMENTS.items():
        d = _load_symbol(sym, cfg, a.score)
        if d is not None:
            data[sym] = d
            print(f"  {sym:<14} {len(d['time'])} bars")
    events = _build_events(data)
    print(f"  {len(events)} total bar-events\n")

    ad = _run(events, data, a.risk, a.cap, True, a.mc, random.Random(7), guard_buffer=a.buffer)
    print(f"  ADAPTIVE  pass {ad['pass_pct']:.1f}%  BLOW {ad['blow_pct']:.2f}%  "
          f"daily {ad['daily_pct']:.1f}%  timeout {ad['timeout_pct']:.1f}%  "
          f"avg_trades {ad['avg_trades']:.1f}  (n={ad['n']})")
    if a.compare:
        fl = _run(events, data, a.risk, a.cap, False, a.mc, random.Random(7))
        print(f"  FLAT      pass {fl['pass_pct']:.1f}%  BLOW {fl['blow_pct']:.2f}%  "
              f"daily {fl['daily_pct']:.1f}%  timeout {fl['timeout_pct']:.1f}%  "
              f"avg_trades {fl['avg_trades']:.1f}")
        print(f"\n  {'Metric':<18}{'FLAT':>10}{'ADAPTIVE':>12}")
        for k, nm in [("pass_pct","Pass %"),("blow_pct","Blow %"),("daily_pct","Daily-DD %"),
                      ("timeout_pct","Timeout %"),("avg_trades","Avg trades")]:
            print(f"  {nm:<18}{fl[k]:>10.2f}{ad[k]:>12.2f}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
