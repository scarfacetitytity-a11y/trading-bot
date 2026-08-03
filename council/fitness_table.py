"""Council persona task-fitness table.

Adapted from OmniRoute open-sse/services/autoCombo/taskFitness.ts —
5-layer resolution collapsed to a static table for AiDEN's deterministic
Council (no LLM, no ELO scraping, scores derived from Council mandate).

fitness[persona_id][task_category] -> float 0.0-1.0
  1.0 = this persona owns this category
  0.0 = irrelevant to this persona

Task categories:
  trade_entry     — should we open a position?
  sizing          — how large?
  risk_management — drawdown, daily limit, position concentration
  compliance      — FTMO rules, trade caps, session windows
  execution_timing— latency, slippage, fill timing
  market_analysis — signal quality, regime, AMD context
  system_health   — process liveness, circuit breakers, retries
"""

FITNESS: dict[str, dict[str, float]] = {
    "01": {  # Principal Engineer — arbitration, blockers, live deployment gate
        "trade_entry":      0.60,
        "sizing":           0.55,
        "risk_management":  0.75,
        "compliance":       0.65,
        "execution_timing": 0.50,
        "market_analysis":  0.30,
        "system_health":    0.90,
    },
    "02": {  # Architect — module boundaries, gate sequence structure
        "trade_entry":      0.30,
        "sizing":           0.35,
        "risk_management":  0.45,
        "compliance":       0.35,
        "execution_timing": 0.30,
        "market_analysis":  0.30,
        "system_health":    0.65,
    },
    "03": {  # Data Engineer — state management, position tracking, equity
        "trade_entry":      0.50,
        "sizing":           0.70,
        "risk_management":  0.75,
        "compliance":       0.50,
        "execution_timing": 0.45,
        "market_analysis":  0.40,
        "system_health":    0.60,
    },
    "04": {  # Security Engineer — credentials, API keys, git exposure
        "trade_entry":      0.10,
        "sizing":           0.10,
        "risk_management":  0.30,
        "compliance":       0.25,
        "execution_timing": 0.15,
        "market_analysis":  0.10,
        "system_health":    0.50,
    },
    "05": {  # Compliance Officer — FTMO rules, daily limits, session windows
        "trade_entry":      0.75,
        "sizing":           0.85,
        "risk_management":  0.95,
        "compliance":       1.00,
        "execution_timing": 0.70,
        "market_analysis":  0.35,
        "system_health":    0.45,
    },
    "06": {  # App Engineer — gate logic, scoring, sizing formula, edge cases
        "trade_entry":      0.95,
        "sizing":           0.90,
        "risk_management":  0.70,
        "compliance":       0.55,
        "execution_timing": 0.80,
        "market_analysis":  0.75,
        "system_health":    0.60,
    },
    "07": {  # SRE — circuit breakers, crash recovery, heartbeat, retries
        "trade_entry":      0.45,
        "sizing":           0.45,
        "risk_management":  0.65,
        "compliance":       0.45,
        "execution_timing": 0.60,
        "market_analysis":  0.25,
        "system_health":    0.95,
    },
    "08": {  # Performance Engineer — execution latency, signal-to-order path
        "trade_entry":      0.65,
        "sizing":           0.50,
        "risk_management":  0.45,
        "compliance":       0.35,
        "execution_timing": 0.95,
        "market_analysis":  0.40,
        "system_health":    0.70,
    },
    "09": {  # Test Engineer — backtest validity, gate path coverage
        "trade_entry":      0.60,
        "sizing":           0.65,
        "risk_management":  0.55,
        "compliance":       0.45,
        "execution_timing": 0.40,
        "market_analysis":  0.80,
        "system_health":    0.40,
    },
    "10": {  # Staff Engineer — blast radius, config duplication, knowledge silos
        "trade_entry":      0.45,
        "sizing":           0.55,
        "risk_management":  0.55,
        "compliance":       0.40,
        "execution_timing": 0.35,
        "market_analysis":  0.40,
        "system_health":    0.65,
    },
    "11": {  # Reality Gap Analyst — backtest vs live divergence
        "trade_entry":      0.85,
        "sizing":           0.70,
        "risk_management":  0.60,
        "compliance":       0.45,
        "execution_timing": 0.50,
        "market_analysis":  0.90,
        "system_health":    0.25,
    },
    "12": {  # Devil's Advocate — all confident conclusions, always last
        "trade_entry":      0.90,
        "sizing":           0.85,
        "risk_management":  0.80,
        "compliance":       0.75,
        "execution_timing": 0.65,
        "market_analysis":  0.85,
        "system_health":    0.60,
    },
}

PERSONA_NAMES = {
    "01": "Principal Engineer",
    "02": "Architect",
    "03": "Data Engineer",
    "04": "Security Engineer",
    "05": "Compliance Officer",
    "06": "App Engineer",
    "07": "SRE",
    "08": "Performance Engineer",
    "09": "Test Engineer",
    "10": "Staff Engineer",
    "11": "Reality Gap Analyst",
    "12": "Devil's Advocate",
}

# Minimum fitness to include in candidate pool for a task category
POOL_THRESHOLD = 0.50

# Personas that always participate regardless of task (governance mandate)
ALWAYS_ACTIVE = {"05", "12"}  # Compliance Officer + Devil's Advocate
