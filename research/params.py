"""
AiDEN autoresearch — editable parameter file.
This is the ONLY file the autoresearch agent modifies.
Adapted from Karpathy's autoresearch train.py pattern.

Current values: baseline from adaptive sweep results (2026-07-26).
"""

# Primary lever — pass/blow master dial
# Adaptive sweep: lower = lower blow rate but more timeouts
BASE_RISK = 0.35  # percent of account per trade

# Daily drawdown halt (percent) — bot stops adding new trades
DAILY_DD_LIMIT = 2.0

# Score threshold (0-100) — only take trades above this
SCORE_FLOOR = 65

# Grade threshold — skip D grades
MIN_GRADE = "C"

# Minimum risk/reward ratio
MIN_RR = 1.5

# Hours to block correlated instruments after a loss
# (US30+US500, US100+US500)
COUSIN_BLOCK_HOURS = 4

# Max simultaneous open positions across all instruments
CAP_CONCURRENT = 3

# Minimum edge ratio (strategy-specific signal strength)
EDGE_FILTER = 1.0

# ATR multiplier for stop loss buffer
SL_BUFFER_ATR = 0.5
