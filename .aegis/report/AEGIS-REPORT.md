# AEGIS Audit Report — trading-bot
**Date:** 2026-07-12  
**Target:** C:\Users\anton\Documents\trading-bot  
**Scope:** Full audit — execution path, risk system, strategy wiring, FTMO compliance

---

## 1. Executive Risk Summary

**Overall risk: HIGH — bot is not safe to run in its current state against an FTMO challenge account.**

Three critical gaps exist that would cause challenge failure or uncontrolled loss before any business logic error occurs:

1. **The RiskAgent psychology gate is not wired into the execution pipeline** — it exists as dead code.
2. **FVG_OB strategy (AiDEN's declared core strategy) is not registered in the Orchestrator** — it cannot be selected via the production runner.
3. **The bot allows short positions** via FVG_OB bear signals — violating the AiDEN long-only constraint.

Everything else is secondary to these three.

---

## 2. Critical Findings (must fix before any live run)

---

### CRIT-01 — RiskAgent not wired into live execution
**Severity:** CRITICAL  
**File:** `execution/live_runner.py`, `execution/orchestrator.py`

`RiskAgent` exists in `execution/risk_agent.py` with full psychology gates (consecutive losses, daily limit, weekly DD, rolling WR check). Neither `live_runner.py` nor `orchestrator.py` imports or calls it.

Every trade placed bypasses:
- Consecutive loss cooldown (3 losses → 24h pause)
- Daily loss soft-limit (4%)
- Weekly DD limit (7%)
- Rolling WR halt threshold (20%)

The only active FTMO guard is `RiskGuard` in the orchestrator (daily 4.5%, total 9.0%) — a hard kill switch that fires after the loss happens. The pre-trade gate that prevents accumulating losses in the first place is absent.

**Fix:** In `orchestrator.py` `TradingEngine.run()`, before calling `trader.place_order()`, call `risk_agent.pre_trade_check(current_equity, open_trade_count)`. Instantiate `RiskAgent` once at orchestrator startup. Call `risk_agent.record_trade()` after each close.

---

### CRIT-02 — FVG_OB not in Orchestrator STRATEGY_MAP
**Severity:** CRITICAL  
**File:** `execution/orchestrator.py:65-68`

```python
STRATEGY_MAP = {
    "sniper":          SniperStrategy,
    "london_breakout": LondonBreakoutStrategy,
}
```

`FVGOrderBlockStrategy` is not imported and not registered. Setting `strategy: fvg_ob` in config.yaml would silently fall back to `SniperStrategy` (the default) with no error.

The declared AiDEN core strategy cannot be selected via the production orchestrator.

**Fix:** Add to `orchestrator.py`:
```python
from strategies.fvg_ob import FVGOrderBlockStrategy
STRATEGY_MAP["fvg_ob"] = FVGOrderBlockStrategy
```

---

### CRIT-03 — FVG_OB generates short signals (violates long-only constraint)
**Severity:** CRITICAL  
**File:** `strategies/fvg_ob.py:193-220`

Bear FVG detection and short entry logic is fully implemented and active. `ob_required=True` (default) doesn't filter direction — it just requires an order block on either side.

AiDEN mandate: **XAUUSD + US30, long-only.** A short signal on XAUUSD during a bear FVG would place a sell order on the FTMO account.

**Fix:** Add `long_only: bool = True` parameter. Skip bear FVG detection and all short position management when enabled. Default to `True` to match AiDEN constraints.

---

## 3. High Findings (fix before extended live session)

---

### HIGH-01 — Config strategy mismatch
**Severity:** HIGH  
**File:** `config/config.yaml:31`

`strategy: forex_master` is set in config. ForexMaster is not the AiDEN strategy. If someone runs `python -m execution.live_runner` without flags, it defaults to `sniper_master` (live_runner default). If they run the orchestrator, it defaults to `sniper`. Neither runs `fvg_ob`.

**Fix:** Set `strategy: fvg_ob` in config.yaml once CRIT-02 is resolved.

---

### HIGH-02 — Symbols include non-AiDEN instruments
**Severity:** HIGH  
**File:** `config/config.yaml:12-15`

```yaml
symbols:
  - XAUUSD
  - US30.cash
  - US100.cash
  - GBPUSD
```

US100.cash and GBPUSD are outside the AiDEN scope. Running the orchestrator with the default config trades 4 symbols with 4 concurrent threads, including two instruments with no validated strategy or risk params.

**Fix:** Remove `US100.cash` and `GBPUSD`. Keep only `XAUUSD` and `US30.cash`.

---

### HIGH-03 — SL/TP uses fixed points, not price-aware sizing
**Severity:** HIGH  
**File:** `execution/risk.py:56-60`, `config/config.yaml:37-38`

```yaml
stop_loss_points: 200   # 20 pips
take_profit_points: 400  # 40 pips
```

`risk.py` applies `sl_points * info.point` regardless of symbol. For XAUUSD, a "point" is $0.01. 200 points = $2 SL on gold — essentially zero protection, or far too tight depending on MT5 broker's point definition.

FVG_OB already computes ATR-based SL internally and stores it in `self._stops`. The orchestrator ignores this and calls `risk.calculate_sl_tp()` with the fixed config points instead.

**Fix:** Pass the strategy's computed SL to the order, not the config fixed points. FVG_OB returns signals+stops already sized. Wire `strategy._stops.iloc[-1]` into `trader.place_order()`.

---

### HIGH-04 — RiskAgent `pause_until` uses naive UTC, comparison bug on restart
**Severity:** HIGH  
**File:** `execution/risk_agent.py:125`

```python
if datetime.utcnow() < pause_dt:
```

`datetime.utcnow()` returns a naive datetime. `datetime.fromisoformat(s.pause_until)` also returns naive (no tz info stored). This works today, but if the bot is moved to a machine with a different timezone or Python version changes, the comparison silently breaks. Use `datetime.now(timezone.utc)` and store ISO strings with `+00:00` suffix.

---

## 4. Medium Findings

---

### MED-01 — live_runner.py and orchestrator.py are parallel execution paths, diverging
**File:** `execution/live_runner.py`, `execution/orchestrator.py`

Two separate live execution implementations. `live_runner.py` has no heartbeat registry, no restart logic, no RiskGuard component. `orchestrator.py` is the production-grade version but has the strategy gap (CRIT-02). Running `live_runner.py` in "production" skips all orchestration safety.

**Fix:** Deprecate `live_runner.py` as a dev/dry-run tool only. All live runs go through the orchestrator.

---

### MED-02 — No FTMO challenge progress tracking
**File:** N/A — missing feature

The bot has no awareness of:
- Current profit target progress (10% = $1,000 on $10k)
- Days remaining in challenge
- Whether the account has crossed the profit threshold

No auto-scaling of aggression as target approaches, no alert when near. AiDEN's pass condition is invisible to the bot.

---

### MED-03 — `allow_real_account: false` — good, but no enforcement test
**File:** `config/config.yaml:41`, `execution/orchestrator.py:574-591`

The safety check exists and blocks real accounts. But there's no test that verifies this check actually fires. A config edit could silently remove the protection.

---

### MED-04 — Backtest engine uses different SL/TP logic than live execution
**File:** `backtests/ftmo_engine.py` vs `execution/risk.py`

Backtest uses strategy-internal SL/TP (correct). Live execution overrides with config fixed points (incorrect per HIGH-03). Backtest results don't reflect live execution behaviour — the reality gap is the SL/TP sizing.

---

## 5. Remediation Roadmap

| Priority | Finding | Effort | Owner |
|----------|---------|--------|-------|
| 1 | CRIT-01: Wire RiskAgent into orchestrator | S (1-2h) | Builder |
| 2 | CRIT-02: Register FVG_OB in STRATEGY_MAP | XS (15min) | Builder |
| 3 | CRIT-03: Add long_only flag to FVG_OB | S (1h) | Builder |
| 4 | HIGH-01: Fix config strategy to fvg_ob | XS (5min) | Builder |
| 5 | HIGH-02: Remove non-AiDEN symbols from config | XS (5min) | Builder |
| 6 | HIGH-03: Wire strategy ATR stops to live execution | M (2-3h) | Builder |
| 7 | HIGH-04: Fix naive datetime in RiskAgent | XS (15min) | Builder |
| 8 | MED-01: Deprecate live_runner.py | XS | Builder |
| 9 | MED-02: Add FTMO progress tracker | M | Builder |

**Sprint 1 (today):** CRIT-01, CRIT-02, CRIT-03, HIGH-01, HIGH-02, HIGH-04 — XS/S items only. Bot goes from dangerous to safe.  
**Sprint 2:** HIGH-03 (SL/TP wiring) + MED-02 (FTMO tracker).

---

## 6. What's Working Well

- Orchestrator architecture is solid — heartbeat registry, kill switch, exponential backoff restart
- RiskGuard hard limits are correctly below FTMO thresholds (4.5%/9.0% vs 5%/10%)
- `allow_real_account: false` blocks accidental live trading
- FVG_OB strategy logic is well-implemented — detection, test, confirmation, invalidation all correct
- ATR-based stop placement inside FVG_OB is sound
- `_clamp_lots()` correctly handles broker min/max/step
- RiskAgent exists and has the right psychology rules — just needs to be connected

---

*AEGIS audit complete — 3 critical, 4 high, 4 medium findings.*
