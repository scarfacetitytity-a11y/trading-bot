# AiDEN architecture map

Audited 2026-10-07 (branch `aiden-self-upgrade-v1`).

## Canonical relationship between the two repos

| Repo | Role | Canonical for |
|---|---|---|
| `scarfacetitytity-a11y/os` (private) | AiDEN Trading Operating System: knowledge, policy, reasoning | Rules and *why* they exist: risk laws, psychology, playbooks, governance, Sniper framework, audit trail (`audits/`). |
| `scarfacetitytity-a11y/trading-bot` (**public**) | Execution, infrastructure, research, automation | *Enforcement*: the numbers the running system obeys (`core/system_invariants.py`), execution, backtests, the self-upgrade lifecycle. |

Rules of the relationship:

1. **Policy flows os → bot.** A risk or governance rule originates in `os`; the bot
   enforces it in `core/system_invariants.py`, whose docstring cites the os file.
2. **When they disagree, the stricter value is enforced and the conflict is
   logged as an open decision** (`docs/RISK_INVARIANTS.md`, D1–D9). Neither side
   silently wins.
3. **Evidence flows bot → os.** Promoted upgrades (`research/upgrades/CHANGELOG.md`)
   and audits are summarised back into `os/audits/`.
4. `trading-bot` is public: no account numbers, personal paths or credentials
   belong in it. Machine-specific locations resolve through `core/paths.py`
   (`AIDEN_VAULT_PATH`, `AIDEN_FIRM_DB`, `AIDEN_FINANCIAL_MCP_DIR`,
   `AIDEN_CLAUDE_CLI`, `AIDEN_LOGS_DIR`).

## Runtime map (trading-bot)

| Flow stage | Component(s) | Status |
|---|---|---|
| Scout / Observer | `execution/scout_reach.py`, `execution/preday_intel.py`, `news/`, `execution/news_gate.py`, Cadre Scout (`cadre_invoke` MEM-004) | Live |
| Market context | `execution/market_context_agent.py`, `execution/level_monitor.py`, `execution/market_phase.py`, `execution/regime_classifier.py`, `execution/scenario_engine.py`, `execution/order_flow.py`, `execution/market_reader.py` | Live (legacy stack) |
| Market context v3 | `core/market_reader.py`, `core/amd_detector.py`, `core/analyzer_engine.py`, `core/signal_lifecycle.py` | Shadow; live per symbol only after cutover validation |
| Strategy engine | `strategies/aiden_index.py` (live), `strategies/sniper.py`, `strategies/fvg_ob.py`, `strategies/london_breakout.py` | |
| Probability | `execution/probability_model.py` (live Bayesian), `core/probability_stack.py` (v3) | Lift updates gated |
| Risk | `core/system_invariants.py` (envelope) → `execution/risk_agent.py` (per-trade psychology/limits) → `RiskGuard` in `orchestrator.py` (equity tiers, kill switch) → `execution/portfolio_manager.py` (allocation) → `execution/ftmo_tracker.py` | |
| Governance | `execution/trade_agent.py`, `council/council_router.py` | |
| Human approval | `telegram_notify.request_approval` via `approval_policy` | Mandatory, fail-closed in live mode |
| Execution | `execution/orchestrator.py` (daemon), `execution/trader.py` | |
| Journal | `execution/trade_journal.py`, `execution/obsidian_sync.py`, `execution/council_obsidian.py` | |
| Outcome analysis | `execution/learning_loop.py`, `execution/trade_analyzer.py`, Cadre Quant | Proposals only |
| Research | `research/` (autoresearch + `upgrade_loop.py`), `backtests/` | |
| Self-upgrade | `core/self_upgrade.py`, `research/upgrade_loop.py`, `execution/upgrade_gate.py` | New |
| Watchdogs | `execution/council_watch.py`, `execution/watchdog.py`, `execution/code_monitor.py`, `council/self_healing.py` | See overlap below |

## Duplication and overlap — decisions

| Overlap | Decision |
|---|---|
| `execution/market_reader.py` vs `core/market_reader.py`; `execution/probability_model.py` vs `core/probability_stack.py` | Intentional during the v3 migration. **Canonical target: `core/` (v3)**, per symbol, once `validate_cutover_ready` passes with ≥ 50 shadow samples. The legacy modules are retired by an upgrade proposal per symbol, not deleted now. |
| `execution/risk.py` (lot clamping + fixed-point SL/TP fallback) vs orchestrator structural sizing | Kept: `_size_order` uses it for broker lot clamping and as the non-ATR fallback. Canonical sizing is `TradingEngine._size_order`; `risk.py` is a helper, not a second policy. |
| `RiskGuard` (orchestrator) vs `RiskAgent` vs `council_watch` thresholds | One envelope (`core/system_invariants.py`); each component reads from or is clamped to it. RiskGuard = primary equity gate; RiskAgent = per-trade behavioural gate; council_watch = out-of-process backstop. |
| `council_halt.flag` (daily, auto-cleared) vs new emergency halt | Both kept: council halt is the automatic daily stop; `emergency_halt.flag` is the persistent human-only stop. |
| `execution/watchdog.py` (process restart), `council_watch.py` (compliance + Cadre scheduler), `council/self_healing.py`, `code_monitor.py` (log scan → Builder) | Kept. **Proposal:** merge `council/self_healing.py` into `watchdog.py` (both are "heal" loops) — not done here because both run in production and need a live soak. |
| `research/backtest_harness.py` vs `research/upgrade_loop.HarnessEvaluator` | Harness kept unchanged as the autoresearch exploration metric; the evaluator reuses the same `run_ftmo_sim_adaptive` engine but adds IS/OOS split, daily halt and stress. |
| `learning_loop` auto-apply vs Quant lift proposals | One path: both write proposals; `upgrade_gate.gate_lift_proposals` decides what reaches the model. |

## Fixed in this branch (summary)

See the PR description for the full list. Highlights: fail-closed human approval,
invariant envelope, pre-order R:R / hard-stop / emergency-halt gate, weekly-DD
wiring bug, profit-lock bypass, learning loop no longer edits source, gated
Bayesian lift updates, clamped Telegram `/risk`, least-privilege Cadre tools,
protected-file integrity check, portable paths, `.env` untracked.

## Upgrade proposals not executed (need a human or a live soak)

1. Retire legacy `execution/market_reader.py` / `probability_model.py` per symbol after v3 cutover evidence.
2. Merge `council/self_healing.py` into `execution/watchdog.py`.
3. Split `execution/orchestrator.py` (~3.9k lines): extract the entry-gate sequence into `execution/entry_gates.py` so each gate is unit-testable without a stubbed engine.
4. Implement the os sizing ladder (D5) in RiskGuard once Jay picks the numbers.
5. Make `research/backtest_harness.py` consume MIN_RR / EDGE_FILTER / SL_BUFFER_ATR so those become measurable.
6. Restructure the `os` repo (duplicate `physcology/` ↔ `# 02_PSYCHOLOGY/`, `goverance/` ↔ `governance/`, flattened `# PART …` paste files) — see `os/architecture/AIDEN_SYSTEM_MAP.md`.
