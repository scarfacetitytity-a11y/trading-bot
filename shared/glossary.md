# AiDEN Glossary

## Signal & Score
| Term | Meaning |
|------|---------|
| `signal` | ±1 (long/short) or 0 (flat) — direction bias from strategy |
| `signal_score` | /10 confluence score — 7 minimum to trade |
| `NO_CONFLUENCE` | Score < 7 — trade skipped before any gate runs |
| AMD sweep | Accumulation-Manipulation-Distribution sweep — +1 or +2 score |
| M5 confirmed | M5 bar close breaks prior M5 high/low in signal direction — +1 score |
| M1 confluence | M1 candle direction aligns with trade — +1 score (NOT a gate) |

## Trade Plan
| Term | Meaning |
|------|---------|
| Grade A | Clean liquidity draw, RR ≥ 2.0, structural SL and TP |
| Grade B | Liquidity draw exists, RR ≥ 1.5 |
| Grade C | No clean draw, ATR fallback target — usually vetoed by TradeAgent |
| RR | Risk/Reward ratio — TP distance / SL distance |
| draw level | Next liquidity pool (prior high/low, FVG, OB) that price is likely to sweep |

## Gate Outcomes
| Term | Meaning |
|------|---------|
| `AGENT GO` | Trade cleared all gates and Council — order placed |
| `NO_GO` | TradeAgent or Council blocked — not sent to MT5 |
| `BLOCKED` | Hard gate blocked (MCAgent, news, daily DD, session) |
| MISFIRE | Trade placed but journal flagged as poor setup profile after review |

## System Components
| Term | Meaning |
|------|---------|
| CousinRouter | Allows only highest-scoring instrument per direction per bar |
| MCAgent | Market Context Agent — evaluates structural level context |
| RiskGuard | Enforces daily DD and max DD limits |
| TradeReconciler | White-blood-cell auto-close if MT5 position vanishes |
| Council of 12 | 12 persona adjudicators — must clear before live order |
| HeartbeatRegistry | Tracks last-alive timestamps for all components |
| watchdog | PowerShell supervisor that auto-restarts crashed orchestrator |
