# FTMO Rules — Hard Constraints

## Phase 1 Parameters
| Rule | Value | Enforcement |
|------|-------|-------------|
| Profit target | 10% ($10,000 on $100k) | Not enforced by bot — manual milestone |
| Max total DD | 10% ($10,000) | RiskGuard hard halt |
| Daily DD limit | 5% of day-start equity | RiskGuard inline gate at every entry |
| Challenge window | 30 calendar days | Day counter in orchestrator |
| Min trading days | 4 | Tracked, not enforced |

## Daily DD Anchor
- `daily_start_equity` is set at midnight UTC (or bot restart, whichever is later)
- Stored in `logs/risk_agent_state.json`
- If daily DD shows 0% but equity is down: check this file — may need manual patch
- Patch command: update `daily_start_equity` to current live equity value

## DD Tier Floors (score gate multipliers)
| DD Level | Effect |
|----------|--------|
| < 1.5% | Normal — no score floor change |
| 1.5–2.5% | Floor +1 (score must exceed floor by +1 more than usual) |
| > 2.5% | Floor +2 |
| ≥ 5.0% | Hard halt — no new trades for remainder of day |

## Hard Rules
- `allow_real_account: false` — NEVER change this. Bot is demo-only.
- No trades during high-impact news ±30min (unless H4 aligned — bypass allowed)
- Session windows: JP225 00-08 UTC, US indices 13-21 UTC, FX 07-17 UTC
- BIDIRECTIONAL on all instruments — never set `long_only=True` anywhere
