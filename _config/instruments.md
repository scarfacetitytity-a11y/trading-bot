# Active Instruments

## Full Universe (15 symbols)
| Symbol | Type | Session (UTC) | Notes |
|--------|------|---------------|-------|
| GBPUSD | FX | 07-17 | |
| EURUSD | FX | 07-17 | |
| AUDUSD | FX | 07-17 | |
| USDJPY | FX | 07-17 | |
| AUDJPY | FX | 07-17 | Asia pair — added 2026-08-04 |
| NZDJPY | FX | 07-17 | Asia pair — added 2026-08-04 |
| GBPJPY | FX | 07-17 | Asia pair — added 2026-08-04 |
| XAUUSD | Metal | 24h | Core instrument — H1 strategy confirmed both directions |
| XAGUSD | Metal | 24h | |
| US100.cash | Index CFD | 13-21 | Cash CFD — not futures |
| US30.cash | Index CFD | 13-21 | Cash CFD — not futures |
| US500.cash | Index CFD | 13-21 | Cash CFD — not futures |
| US2000.cash | Index CFD | 13-21 | Cash CFD — not futures |
| UK100.cash | Index CFD | 07-17 | |
| JP225.cash | Index CFD | 00-08 | |

## Key Rules
- ALL instruments are BIDIRECTIONAL — long and short allowed on every symbol
- `BIDIRECTIONAL` set must equal `set(INSTRUMENTS.keys())` — verified 2026-07-23
- CousinRouter selects highest-scoring instrument per direction per bar
