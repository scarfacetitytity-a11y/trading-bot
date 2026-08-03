"""Council composite scorer.

Direct port of OmniRoute open-sse/services/autoCombo/scoring.ts
calculateScore() / calculateFactors() / scorePool() — translated to Python.

Factors (weights sum to 1.0):
  task_fit         0.35  persona fitness for this decision category
  recent_accuracy  0.25  fraction of this persona's past verdicts that
                         matched actual outcomes (from learning loop data)
  da_weight        0.20  Devil's Advocate gets extra weight when near FTMO
                         limits (the closer to daily limit, the higher)
  urgency_match    0.10  execution-speed personas rank higher for time-critical
  stability        0.10  1 - error_rate for this persona recently
"""
from __future__ import annotations

from .fitness_table import FITNESS, PERSONA_NAMES

# Default weight profile — overridden by strategy_registry
DEFAULT_WEIGHTS: dict[str, float] = {
    "task_fit":        0.35,
    "recent_accuracy": 0.25,
    "da_weight":       0.20,
    "urgency_match":   0.10,
    "stability":       0.10,
}

# Urgency-sensitive personas (score higher in time-critical decisions)
_URGENCY_PERSONAS = {"06", "08", "01"}


def calculate_factors(
    persona_id: str,
    task_category: str,
    context: dict,
    accuracy_map: dict[str, float],
    error_rate_map: dict[str, float],
) -> dict[str, float]:
    """Compute raw factor values [0,1] for one persona candidate.

    context keys used:
      daily_dd_pct    float  current daily drawdown %
      is_urgent       bool   time-critical decision (pre-news, near session end)
      da_boost        float  0-1 extra weight for DA (caller sets based on stakes)
    """
    task_fit = FITNESS.get(persona_id, {}).get(task_category, 0.0)

    recent_accuracy = accuracy_map.get(persona_id, 0.50)  # prior = coin flip

    # Devil's Advocate gets boosted as daily DD approaches FTMO limit
    if persona_id == "12":
        dd_pct = context.get("daily_dd_pct", 0.0)
        da_boost = min(dd_pct / 5.0, 1.0)          # 0% DD → 0.0, 5% DD → 1.0
        da_factor = 0.50 + 0.50 * da_boost          # 0.50 baseline → 1.0 at limit
    else:
        da_factor = 0.0

    urgency_match = 1.0 if (context.get("is_urgent") and persona_id in _URGENCY_PERSONAS) else 0.5

    stability = 1.0 - error_rate_map.get(persona_id, 0.0)

    return {
        "task_fit":        task_fit,
        "recent_accuracy": recent_accuracy,
        "da_weight":       da_factor,
        "urgency_match":   urgency_match,
        "stability":       stability,
    }


def calculate_score(factors: dict[str, float], weights: dict[str, float]) -> float:
    """Dot product of factor values × weights, clamped [0,1]."""
    raw = sum(factors[k] * weights.get(k, 0.0) for k in factors)
    return max(0.0, min(1.0, raw))


def score_pool(
    candidates: list[str],
    task_category: str,
    context: dict,
    accuracy_map: dict[str, float],
    error_rate_map: dict[str, float],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Score all candidates and return sorted list (highest first).

    Returns list of:
        {persona_id, name, score, factors}
    """
    w = weights or DEFAULT_WEIGHTS
    results = []
    for pid in candidates:
        factors = calculate_factors(pid, task_category, context, accuracy_map, error_rate_map)
        score   = calculate_score(factors, w)
        results.append({
            "persona_id": pid,
            "name":       PERSONA_NAMES.get(pid, f"#{pid}"),
            "score":      round(score, 4),
            "factors":    {k: round(v, 3) for k, v in factors.items()},
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
