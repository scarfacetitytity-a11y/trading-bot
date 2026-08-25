# Risk Parameters

## Sizing
| Param | Value | Location |
|-------|-------|----------|
| `base_risk` | Set in config.yaml | Master risk dial — drives pass/blow rate most |
| `risk_pct` | Fixed-fractional per trade | Percentage of current equity |
| Grade A size | 1.0× | Full size |
| Grade B size | 0.75× | Standard |
| Grade C size | 0.25× | Low conviction — often NO_GO from TradeAgent |

## Trade Management Thresholds
| Param | Value | Notes |
|-------|-------|-------|
| T1 partial % | 0.5 (50%) | `t1_partial_pct` — was causing signal truncation bug (fixed 2026-08-03) |
| T1 target | Varies by plan | First partial take at T1 draw level |
| Profit-lock SL | Entry+buffer | Placed on partial close |
| Trail distance | ATR-based | H4 structure aware |

## Sweep Parameters (secondary optima from sweep)
| Param | Value |
|-------|-------|
| cap | 4 |
| buf | 0.5 |
| edge | 1 |
| min | 0.5 |

## Notes
- `base_risk` is the primary pass/blow dial — lower risk trades daily-DD timeout for blow rate reduction
- Stage-2 joint sweep pending — honest pass rate ~40% (not 90%)
- 5% daily limit fails ~32% of challenges at standard risk
