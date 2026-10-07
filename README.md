# AiDEN Trading Bot — FTMO Prop-Firm Execution System

Autonomous Python/MT5 trading system for FTMO challenges. FVG + Order Block
confluence on M15/H1, fully bidirectional across forex, metals, and index CFDs,
governed by a Bayesian probability model, an active market-context agent, and
an always-on compliance daemon.

**Demo accounts only.** `trading.allow_real_account: false` is a hard invariant.

**Human approval is mandatory by default.** `execution_mode: live` (default) means
every order needs a Telegram APPROVE and fails closed. See
[`docs/RISK_INVARIANTS.md`](docs/RISK_INVARIANTS.md),
[`docs/SELF_UPGRADE.md`](docs/SELF_UPGRADE.md),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Canonical knowledge/policy lives in the private `os` repo; this repo enforces it.

## Architecture

```
trading-bot/
├── core/
│   ├── system_invariants.py     # PROTECTED: immutable risk/approval envelope
│   ├── self_upgrade.py          # PROTECTED: proposal lifecycle + evidence bar
│   ├── paths.py                 # portable machine paths (AIDEN_* env vars)
│   └── ...                      # v3 stack (analyzer_engine, probability_stack)
├── execution/
│   ├── upgrade_gate.py          # PROTECTED: what may reach live execution
│   ├── orchestrator.py          # main daemon: per-symbol engines, RiskGuard, reconciler
│   ├── probability_model.py     # Bayesian P(win) + quarter-Kelly sizing, live-updated lifts
│   ├── market_context_agent.py  # HTF level intelligence: blocks entries fighting key levels
│   ├── trade_analyzer.py        # structural stops/targets, post-trade thesis validation
│   ├── trade_manager.py         # in-trade management (trail, partials, counter-signals)
│   ├── risk_agent.py            # per-trade risk config + circuit breakers
│   ├── ftmo_tracker.py          # challenge progress, daily/total DD anchored to live equity
│   ├── council_watch.py         # standalone enforcement daemon + Cadre agent scheduler
│   ├── cadre_invoke.py          # Sage/Quant/Builder/Scout headless invocations (claude -p)
│   ├── level_monitor.py         # weekly/daily/session key levels
│   ├── order_flow.py            # tick-volume delta, DOM walls (bookmap-lite)
│   └── news_gate.py             # macro event windows
├── strategies/                  # AiDEN Index v2 (FVG+OB confluence scoring)
├── backtests/                   # engine + multi-instrument runners
├── research/                    # autoresearch + upgrade_loop.py, upgrades/ ledger
├── tests/                       # pytest suite (gates, tracker, risk, portfolio)
├── config/config.example.yaml   # template — copy to config.yaml (gitignored)
└── scripts/                     # setup, VPS bootstrap, watchdog, utilities
```

## Entry defense layers

Every entry passes ~14 sequential gates: council halt flag → DD circuit breakers
→ FTMO limits → market-context block → continuation gates → Bayesian EV gate →
portfolio risk cap → min-stop floor (0.75 ATR) → no naked orders → burned-target
guard (90 min) → duplicate guard → liquidity thesis (NO-DRAW) → per-trade agent
verdict → system invariants (emergency halt, hard stop, R:R ≥ 1.5, risk ceiling)
→ human approval (mandatory in live mode). Sizing is quarter-Kelly from P(win),
scaled by context lift and concentration, bounded by a 4% portfolio cap.

## Cadre agent loops (autonomous)

| Agent | Cadence | Output |
|-------|---------|--------|
| Scout | 30 min | `logs/cadre_regime_state.json` → DXY/regime alignment into entry confluences |
| Quant | 2 h + every trade close | `logs/quant_lift_proposals.json` → lift corrections, applied only via the upgrade gate (n ≥ 30) |
| Builder | hourly error scan | targeted code fixes; may not touch protected safety files |
| Sage | Mon 07:00 UTC | weekly strategy/regime review |

## Running

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config\config.example.yaml config\config.yaml   # then fill in MT5 details

# Main bot (PID-locked — refuses to double-start):
python -m execution.orchestrator

# Compliance daemon (run beside the bot, separate process):
python -m execution.council_watch          # or --once for a single health check

# Tests:
pytest
```

## Backtesting

```powershell
python -m backtests.run_compare                    # all strategies × symbols × years
python -m backtests.run_multi_instrument           # FTMO multi-engine simulation
```

## Key runtime state (logs/, gitignored)

- `ftmo_tracker_state.json` — challenge progress; `day_start_equity` anchors daily DD
- `prob_model_state.json` — live Bayesian lift table
- `burned_targets.json` — 90-min re-entry blocks per direction+target
- `council_halt.flag` — daily external halt; orchestrator refuses new entries while present
- `emergency_halt.flag` — persistent human-only halt (`python -m execution.upgrade_gate --halt/--clear-halt`)
- `trades.jsonl` — full trade journal with thesis validation and lessons
