# Risk invariants — audit and unified envelope

Source of truth for enforcement: `core/system_invariants.py` (protected file).
Source of truth for *policy and reasoning*: the `os` repo (`risk/`, `governance/`).
This document records how the two were reconciled on 2026-10-07 (branch
`aiden-self-upgrade-v1`) and what is still waiting for a human decision.

## The model

- **Envelope, not operating values.** Invariants are the outer bounds. Config
  (`config_vps.yaml`) may set equal or *tighter* values. Anything looser is
  clamped at startup and logged as `[INVARIANT]` at CRITICAL.
- **Clamp, don't refuse.** A refused start would leave open positions without
  trade management, so the envelope clamps and the bot keeps running.
- **Nothing automatic may loosen a risk limit.** Not autoresearch, not Quant,
  not the learning loop, not Telegram `/risk`, not an agent. Loosening is only
  possible by a human editing config inside the envelope; widening the envelope
  itself means editing a protected file in a reviewed PR.

## What was found (before this branch)

| Limit | os: live-trading-roles | os: risk/daily_limits | RiskAgent default | orchestrator const | config_vps | council_watch | research/params |
|---|---|---|---|---|---|---|---|
| Daily loss halt | **2.0%** | stop at **3.75%** (75% of 5%), halve size at 2.5% | 2.0% | **4.0%** | 4.0% | warn 3.5 / hard **4.5%** | 2.0% (unused by harness) |
| Weekly DD | — | — | 7% | — | — (RiskAgent fed `soft_dd_halt_pct` = **9%**) | — | — |
| Soft DD (no new entries) | 7% | min size at 9%, halve risk at 8% | — | 7% | **9%** | warn 7 / hard 9 | — |
| Kill / flatten | 10% | 10% = fail | 7% (account, from peak) | 9.5% | 9.5% | — | — |
| Risk per trade | — | "1–2% likely"; preferred 0.5% | — | default **1.0%** | 0.75% | — | 0.35% |
| Concurrent trades | — | TBD | **20** (no cap) | via 4% portfolio cap | — | — | 3 |
| Min R:R | — | **1.5 hard floor** | — | **not enforced** pre-order | — | — | 1.5 (unused by harness) |
| Human approval | Telegram gate described | — | — | `require_approval` default **False**; no Telegram ⇒ approve; timeout ⇒ auto-approve if score ≥ 6 | not set | — | — |
| v3 cutover samples | — | — | — | default 50 | **0** (bypass) | — | — |

Other unsafe paths found:

- `learning_loop.auto_apply` regex-rewrote `execution/trade_analyzer.py` in place,
  contradicting its own "HARD RULE: never edits strategy or config".
- Bayesian lift proposals went live from 3 trades (learning loop) or 20 trades
  (Quant routine), bounded only by 0.5x–2x.
- Telegram `/risk N` set any risk %, unbounded.
- Every Cadre agent was launched with `Read,Write,Edit,Glob,Grep,Bash,WebSearch,WebFetch`;
  Scout (which reads untrusted web content) could run shell commands.
- Builder could autonomously edit risk code with no gate.
- `trade_manager` profit-lock ladder was bypassed by earlier `TIGHTEN_SL` /
  `EXTEND_TP` / `HOLD_RUNNER` branches — a trade that reached 0.9R could get a
  stop *below* break-even (`test_profit_lock_ladder` had been failing).
- `council_halt.flag` is auto-cleared every new day — there was no persistent,
  human-only emergency halt.

## The envelope (what is enforced now)

| Invariant | Value | Reasoning |
|---|---|---|
| `max_daily_loss_pct` | 4.0 | Strictest value every running component already honours; 1% buffer to FTMO 5%. Backstop (council_watch) now uses the same value instead of 4.5. |
| `max_weekly_dd_pct` | 7.0 | RiskAgent's own documented limit; wiring bug fixed (new key `weekly_dd_halt_pct`). |
| `max_soft_dd_pct` | 9.0 | os: at 90% of max DD trade minimum size only. |
| `max_total_kill_pct` | 9.5 | Flatten before FTMO 10%. |
| `max_risk_per_trade_pct` | 1.0 | Existing code default; config runs 0.75; Telegram `/risk` clamps here. |
| `max_portfolio_risk_pct` | 4.0 | Existing portfolio cap. |
| `max_concurrent_trades` | 8 | 4.0% portfolio / 0.5% min trade risk — non-binding for current config; replaces "20". |
| `max_daily_entries` | 6 | Existing RiskAgent cap. |
| `max_consecutive_losses_before_pause` | 7 | Existing RiskAgent value. |
| `min_reward_risk` | 1.5 | os hard floor — now checked before every order (when a TP exists). |
| `min_cutover_samples` | 50 | Closes the `go_no_go_min_samples: 0` bypass. |
| `min_lift_proposal_trades` | 30 | Devil's Advocate minimum already used by the learning loop. |
| `require_hard_stop` | true | Stop must exist and be on the correct side at placement. |
| `require_human_approval_live` | true | `execution_mode: live` (default) ⇒ approval mandatory and fail-closed. |
| Emergency halt | `logs/emergency_halt.flag` or `AIDEN_EMERGENCY_HALT=1` | Persistent; only `python -m execution.upgrade_gate --clear-halt` removes it. |

### Human approval — behaviour change

FTMO challenge and funded accounts report as MT5 **demo** accounts, so
`allow_real_account` alone does not mean "no capital at risk". From this branch:

- `execution_mode: live` (the default, and set explicitly in `config_vps.yaml`):
  every order needs a Telegram APPROVE. No Telegram, a failed send or a timeout is a **veto**.
- `execution_mode: demo_autonomous`: must be written explicitly; restores the old
  behaviour (optional approval, score-based auto-approve). Ignored on a real account.

## Open decisions (need Jay — not chosen automatically)

| # | Decision | Options | Note |
|---|---|---|---|
| D1 | Daily halt level | 2.0 (os live-trading-roles) · 3.75 (os daily_limits) · 4.0 (current) | Lowering the envelope is always allowed; the two os docs also disagree with each other. |
| D2 | Concurrent trade cap | 8 (envelope) · 3 (autoresearch baseline) | |
| D3 | Loss-streak / low-WR response | graduated size cut (current code) · hard pause at 3 losses / WR ≤ 20% (old tests) | Tests now assert the current graduated behaviour and that size never increases after a loss. |
| D4 | Risk per trade | 0.75% (config) · 0.5% (stated preference) | |
| D5 | os sizing ladder (halve at 50% of daily, stop at 75%; halve at 80% DD, min size at 90%) | implement in RiskGuard · keep score-floor tiers | Code uses score-floor tiers 1.5/2.5/3.5%. |
| D6 | Min R:R | 1.5 floor (enforced) · 2.0 (preferred target) | |
| D7 | News window | os: no trading ±5 min FOMC/NFP/CPI, no holding through major news · `_config/ftmo-rules.md`: ±30 min **with H4-aligned bypass** | The bypass conflicts with "no uncontrolled news gambling". |
| D8 | Direction | os `trading-context.md` / `.paul/STATE.md`: long-only · bot: "BIDIRECTIONAL — never long_only" | os docs look stale. |
| D9 | Account size / FTMO rules | os: $10k, 10 min trading days · `config_vps`: `initial_equity: 100000`, `_config/ftmo-rules.md`: 4 min days | Verify against the live FTMO account. |
