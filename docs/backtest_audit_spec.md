---
type: spec
status: draft
tags: [backtest, ftmo, risk, reality-gap, trade-analyzer]
relatedTo: [execution/trade_analyzer.py, backtests/tm_backtest.py, backtests/audit_naive.py, backtests/ftmo_montecarlo.py]
---

# Backtest Audit & Per-Trade Agent Spec

Author: AiDEN execution partner · 2026-07-16 · Council: #09 Test, #11 Reality Gap, #12 Devil's Advocate

## 1. Confirmed flaws in the current backtests

### 1.1 Exit resolution is M15-granular (the "stop-out counted as a win" family)
`tm_backtest.py` and `audit_naive.py` both walk **M15 bars** and detect SL/TP on
that bar's high/low, assuming **SL-before-TP** within a bar. Two consequences:
- **Conservative but wrong.** A single M15 bar that spans both SL and TP is
  *guessed* as SL-first. With structural stops now as tight as 0.5×ATR, both
  levels frequently sit inside one M15 bar — so the guess fires constantly and is
  right only ~50% of the time. We do not know the true intrabar path.
- **The real "stop logged as win" is the T1 labelling trick** (already found): a
  trade tags +1R, books 50%, reverses to breakeven, logs pnl>0 → counted a win.
  This inflated 38% → 72% WR. Fixed in the naive audit; still present in the TM stack.

**Fix:** resolve every trade's SL/TP/partial/trail on the **finest available bars**
(M5 now, M1 preferred). Walk M5/M1 bars inside each holding window and take the
level that is touched **first in chronological order** — no guessing.

### 1.2 Daily drawdown can exceed 5% in the sims — it should be impossible live
The Monte Carlo (`ftmo_montecarlo.py`) applies a whole day's batch of trades even
after the account is down >5% that day. Live, the −2% halt stops new entries and
the −5% is a **hard floor**. So sim daily-DD tails are not achievable by the live
system. **Fix:** model the exact live daily logic — halt new entries at −2%, and
treat −5% as a hard close-all floor. Then daily DD is capped by construction and
any excess in a report is a bug, exactly as Anton says.
Caveat: already-open concurrent positions can still bleed past the halt to their
stops — so the true cap is `halt% + (open positions' remaining risk)`. This is why
the **correlation cluster cap** matters: it bounds how much open risk can stack.

### 1.3 Per-trade risk is no longer a flat % — it must be measured, not assumed
Fixed-fractional sizing already handles variable structural stop distance (lots =
risk% / stop_distance). The new problem is **aggregate/portfolio heat**: N
concurrent trades each at X% = up to N·X% open risk. The backtest currently never
tracks simultaneous open risk. **Fix:** log `risk_at_open` per trade and a running
`portfolio_heat` (sum of open trades' remaining risk). Size caps should key off heat.

### 1.4 No spread / slippage modelling
Commission 0.0001 only. Tight structural stops are **spread-sensitive** — a 0.5×ATR
stop on a 1.5-spread instrument is materially affected. **Fix:** subtract half-spread
on entry and exit (and model stop slippage) from per-symbol typical spreads.

## 2. Missing backtest data / metrics (the "everything" list)

Per-trade (one row each, intrabar-resolved):
- entry/exit time, holding bars, symbol, direction, **trade_type, grade**
- entry, structural stop, liquidity target, exit price, **exit reason** (SL/TP/BE/TM/trail)
- **R risked** (structural), **R realized**, MFE_R, MAE_R, hit_target(bool)
- risk_at_open %, spread paid, portfolio_heat at open

Aggregate:
- total trades, trades/week, **true WR** (hit-TP based, not T1-labelled), profit factor
- avg win R, avg loss R, expectancy R, Sharpe, Sortino
- max DD, **max DAILY DD**, avg daily DD, longest losing streak, worst day/week R

Prop-firm specific (per risk level):
- **pass rate**, days-to-pass distribution (median + p10/p90), **blow rate** (daily vs total split)
- **payout rate**, time-to-first-payout, expected #challenges to funded
- max concurrent open risk (heat), correlation-cluster frequency

Regime breakdowns:
- by session (Asia/London/NY), by instrument, by month, by volatility regime, by trade_type/grade

## 3. Data we may need to pull
- **M1 bars** for all instruments — required for honest tight-stop intrabar resolution.
- **Typical spread per symbol** (and time-of-day spread) — for cost modelling.
- Optionally tick data for the highest-conviction instruments (gold) for exactness.

## 4. Per-trade agent layer (design)

Goal: every candidate trade is adjudicated individually, producing a durable
GO/NO-GO ticket with the thesis and the relevant Council voices — not a silent
arithmetic decision. Two surfaces already exist to build on: the `trade-check`
skill (pre-trade gate → GO ticket / NO TRADE) and the analyzer (`analyze_entry`).

**TradeAgent (per trade):**
1. Runs `analyze_entry` → structural stop, liquidity target, type, grade, size.
2. Runs the gate sequence (M5 trigger, accumulation, liquidity draw, correlation
   divergence + cluster cap, spread check, portfolio heat).
3. Invokes the **relevant Council members** for this trade type:
   - #05 Compliance (daily/total DD room), #06 App (sizing/edge cases),
     #08 Perf (entry timing), #11 Reality Gap (does the drawn liquidity exist),
     #12 Devil's Advocate (what kills this trade).
4. Emits a **TradeTicket**: GO/NO-GO, grade, plan (SL/TP/size), thesis, dissent notes.
5. On close, `analyze_exit` writes the post-mortem back to the ticket → learning loop.

Implementation options:
- **Within Council** — TradeAgent as a thin orchestrator calling existing Council
  personas per trade (keeps one governance source).
- **Separate agent** — a dedicated `trade_agent.py` that owns the ticket lifecycle
  and calls the analyzer + Council. Cleaner blast radius; recommended.

## 5. Recommended build order
1. **M5-resolved backtest engine** (fix 1.1) — the foundation; every metric depends
   on honest exits. Pull M1 next for the tightest stops.
2. **Exact daily-halt + hard-floor + portfolio-heat** in the sim (fix 1.2, 1.3).
3. **Spread/slippage** modelling (fix 1.4).
4. **Full metrics report** (section 2) — one command → per-trade CSV + summary + charts.
5. **Per-trade agent + ticket** (section 4) — once the backtest can score its decisions.
