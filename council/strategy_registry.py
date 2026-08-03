"""Strategy registry — named weight profiles per FTMO market state.

Port of OmniRoute open-sse/services/autoCombo/modePacks.ts pattern.
Each profile is a weight dict for the composite scorer. Profiles switch
automatically via select_strategy() based on live FTMO state.

Profiles:
  normal_market    — standard weights, balanced scoring
  high_volatility  — risk + DA get more weight; compliance gates tighter
  near_daily_limit — compliance dominates; everything else subordinate
  recovery_mode    — maximum risk aversion; only best setups allowed
  pre_news         — compliance + execution speed prioritised
"""
from __future__ import annotations

# Profile weight dicts — keys match scorer.DEFAULT_WEIGHTS
PROFILES: dict[str, dict[str, float]] = {
    "normal_market": {
        "task_fit":        0.35,
        "recent_accuracy": 0.25,
        "da_weight":       0.20,
        "urgency_match":   0.10,
        "stability":       0.10,
    },
    "high_volatility": {
        "task_fit":        0.28,
        "recent_accuracy": 0.20,
        "da_weight":       0.30,   # DA gets more weight
        "urgency_match":   0.12,
        "stability":       0.10,
    },
    "near_daily_limit": {
        "task_fit":        0.15,   # task specialisation matters less
        "recent_accuracy": 0.10,
        "da_weight":       0.20,
        "urgency_match":   0.05,
        "stability":       0.50,   # stability = compliance accuracy dominates
    },
    "recovery_mode": {
        "task_fit":        0.15,
        "recent_accuracy": 0.10,
        "da_weight":       0.35,
        "urgency_match":   0.05,
        "stability":       0.35,
    },
    "pre_news": {
        "task_fit":        0.30,
        "recent_accuracy": 0.20,
        "da_weight":       0.20,
        "urgency_match":   0.20,   # execution speed matters near news
        "stability":       0.10,
    },
}


def select_strategy(daily_dd_pct: float, atr_ratio: float = 1.0,
                    news_window: bool = False) -> str:
    """Pick the active weight profile from live FTMO state.

    daily_dd_pct: current intraday drawdown as percent of equity (0-5%)
    atr_ratio:    current ATR / 20-day avg ATR (>1.5 = elevated volatility)
    news_window:  True if a high-impact news event fires in the next 30 min
    """
    if daily_dd_pct >= 3.5:
        return "near_daily_limit"
    if daily_dd_pct >= 2.5:
        return "recovery_mode"
    if news_window:
        return "pre_news"
    if atr_ratio >= 1.5:
        return "high_volatility"
    return "normal_market"


def get_weights(strategy_name: str) -> dict[str, float]:
    return PROFILES.get(strategy_name, PROFILES["normal_market"])
