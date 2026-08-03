"""Council of 12 governance module.

OmniRoute-adapted multi-persona routing for AiDEN trade decisions.

Public interface:
    CouncilRouter   — main entry point; call .governance_check() per trade
    CouncilVerdict  — return type from governance_check()
    SelfHealingManager.get() — persona health singleton
"""
from .council_router import CouncilRouter, CouncilVerdict
from .self_healing import SelfHealingManager
from .strategy_registry import select_strategy, get_weights, PROFILES

__all__ = [
    "CouncilRouter",
    "CouncilVerdict",
    "SelfHealingManager",
    "select_strategy",
    "get_weights",
    "PROFILES",
]
