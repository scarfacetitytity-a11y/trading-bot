"""Council self-healing manager.

Port of OmniRoute open-sse/services/autoCombo/selfHealing.ts
getSelfHealingManager() singleton — circuit breaker per Council persona.

States: ACTIVE → DEGRADED → EXCLUDED → PROBE → ACTIVE
  ACTIVE:   persona in normal rotation
  DEGRADED: accuracy below threshold — weight penalty applied
  EXCLUDED: too many errors — pulled from candidate pool with backoff
  PROBE:    one-shot test after backoff expires; recovery on success

Persistence: logs/council_health.json — survives process restarts.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent.parent / "logs" / "council_health.json"

# Thresholds
_DEGRADED_THRESHOLD  = 0.35  # accuracy below this → DEGRADED
_EXCLUDED_THRESHOLD  = 0.20  # accuracy below this → EXCLUDED
_MIN_DECISIONS       = 5     # need this many before penalising
_BASE_BACKOFF_SECS   = 300   # 5 min base backoff when EXCLUDED
_MAX_BACKOFF_SECS    = 1800  # 30 min cap

_STATES = ("ACTIVE", "DEGRADED", "EXCLUDED", "PROBE")


class PersonaHealth:
    def __init__(self, persona_id: str):
        self.persona_id    = persona_id
        self.state         = "ACTIVE"
        self.n_decisions   = 0
        self.n_correct     = 0
        self.error_rate    = 0.0
        self.backoff_count = 0
        self.excluded_at   = 0.0

    def accuracy(self) -> float:
        if self.n_decisions < _MIN_DECISIONS:
            return 0.50   # neutral prior
        return self.n_correct / self.n_decisions

    def record_outcome(self, correct: bool) -> None:
        self.n_decisions += 1
        if correct:
            self.n_correct += 1
        acc = self.accuracy()
        if self.state in ("ACTIVE", "DEGRADED"):
            if acc < _EXCLUDED_THRESHOLD and self.n_decisions >= _MIN_DECISIONS:
                self._exclude()
            elif acc < _DEGRADED_THRESHOLD and self.n_decisions >= _MIN_DECISIONS:
                self.state = "DEGRADED"
                self.error_rate = 1.0 - acc
            else:
                self.state     = "ACTIVE"
                self.error_rate = max(0.0, 1.0 - acc)
        elif self.state == "PROBE":
            if correct:
                logger.info("[CouncilHealth] Persona #%s recovered → ACTIVE", self.persona_id)
                self.state         = "ACTIVE"
                self.backoff_count = 0
                self.error_rate    = max(0.0, 1.0 - acc)
            else:
                self._exclude()

    def _exclude(self) -> None:
        self.state       = "EXCLUDED"
        self.excluded_at = time.time()
        backoff          = min(_BASE_BACKOFF_SECS * (2 ** self.backoff_count), _MAX_BACKOFF_SECS)
        self.backoff_count += 1
        logger.warning(
            "[CouncilHealth] Persona #%s EXCLUDED (acc=%.0f%%) — backoff %ds",
            self.persona_id, self.accuracy() * 100, backoff,
        )

    def backoff_remaining(self) -> float:
        backoff = min(_BASE_BACKOFF_SECS * (2 ** max(0, self.backoff_count - 1)), _MAX_BACKOFF_SECS)
        return max(0.0, (self.excluded_at + backoff) - time.time())

    def is_available(self) -> bool:
        if self.state == "ACTIVE":
            return True
        if self.state == "DEGRADED":
            return True   # included with penalty weight
        if self.state == "EXCLUDED":
            if self.backoff_remaining() <= 0:
                self.state = "PROBE"
                logger.info("[CouncilHealth] Persona #%s → PROBE (backoff expired)", self.persona_id)
                return True   # allow one probe
            return False
        if self.state == "PROBE":
            return True
        return True

    def weight_penalty(self) -> float:
        """Multiplier [0,1] applied to composite score. 1.0 = no penalty."""
        if self.state == "DEGRADED":
            return 0.60
        if self.state == "PROBE":
            return 0.70
        return 1.0

    def to_dict(self) -> dict:
        return {
            "state": self.state, "n_decisions": self.n_decisions,
            "n_correct": self.n_correct, "error_rate": round(self.error_rate, 4),
            "backoff_count": self.backoff_count, "excluded_at": self.excluded_at,
        }

    @classmethod
    def from_dict(cls, persona_id: str, d: dict) -> "PersonaHealth":
        h = cls(persona_id)
        h.state         = d.get("state", "ACTIVE")
        h.n_decisions   = d.get("n_decisions", 0)
        h.n_correct     = d.get("n_correct", 0)
        h.error_rate    = d.get("error_rate", 0.0)
        h.backoff_count = d.get("backoff_count", 0)
        h.excluded_at   = d.get("excluded_at", 0.0)
        return h


class SelfHealingManager:
    """Singleton. Thread-safe via simple dict reads (GIL sufficient here)."""

    _instance: "SelfHealingManager | None" = None

    def __init__(self) -> None:
        self._health: dict[str, PersonaHealth] = {}
        self._load()

    @classmethod
    def get(cls) -> "SelfHealingManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_health(self, persona_id: str) -> PersonaHealth:
        if persona_id not in self._health:
            self._health[persona_id] = PersonaHealth(persona_id)
        return self._health[persona_id]

    def is_available(self, persona_id: str) -> bool:
        return self.get_health(persona_id).is_available()

    def weight_penalty(self, persona_id: str) -> float:
        return self.get_health(persona_id).weight_penalty()

    def record_outcome(self, persona_id: str, correct: bool) -> None:
        self.get_health(persona_id).record_outcome(correct)
        self._save()

    def accuracy_map(self) -> dict[str, float]:
        return {pid: h.accuracy() for pid, h in self._health.items()}

    def error_rate_map(self) -> dict[str, float]:
        return {pid: h.error_rate for pid, h in self._health.items()}

    def report(self) -> str:
        lines = ["Council health:"]
        for pid, h in sorted(self._health.items()):
            lines.append(
                f"  #{pid} {h.state:<10} acc={h.accuracy():.0%} "
                f"n={h.n_decisions} penalty={h.weight_penalty():.2f}"
            )
        return "\n".join(lines)

    def _load(self) -> None:
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                for pid, d in data.items():
                    self._health[pid] = PersonaHealth.from_dict(pid, d)
        except Exception as exc:
            logger.debug("[CouncilHealth] load failed: %s", exc)

    def _save(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(exist_ok=True)
            _STATE_FILE.write_text(
                json.dumps({pid: h.to_dict() for pid, h in self._health.items()}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("[CouncilHealth] save failed: %s", exc)
