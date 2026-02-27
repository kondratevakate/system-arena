"""Policy decision result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from arena.core.types import Action


@dataclass
class PolicyResult:
    """
    The result of a policy decision.

    Contains the chosen action and optional metadata about the decision.
    """

    action: Action
    confidence: float | None = None  # 0.0 to 1.0
    reasoning: str | None = None  # Explanation for the decision
    alternatives: list[Action] = field(default_factory=list)  # Other considered actions
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timing
    decision_time_ms: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action": str(self.action),
            "action_type": self.action.action_type,
            "action_params": self.action.params,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": [str(a) for a in self.alternatives],
            "metadata": self.metadata,
            "decision_time_ms": self.decision_time_ms,
        }


@dataclass
class ConstraintViolation:
    """A policy constraint that was violated."""

    constraint_name: str
    severity: str  # "warning", "violation", "critical"
    message: str
    action: Action
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.constraint_name}: {self.message}"


@dataclass
class PolicyDecision:
    """
    Complete record of a policy decision, including validation.

    This is what gets recorded in the execution trace.
    """

    # Input
    decision_point_id: str
    timestamp: datetime

    # Output
    result: PolicyResult
    violations: list[ConstraintViolation] = field(default_factory=list)

    # Policy metadata
    policy_name: str = ""
    policy_version: str = ""

    @property
    def has_violations(self) -> bool:
        """Check if any constraints were violated."""
        return len(self.violations) > 0

    @property
    def has_critical_violations(self) -> bool:
        """Check if any critical constraints were violated."""
        return any(v.severity == "critical" for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision_point_id": self.decision_point_id,
            "timestamp": self.timestamp.isoformat(),
            "result": self.result.to_dict(),
            "violations": [
                {
                    "constraint_name": v.constraint_name,
                    "severity": v.severity,
                    "message": v.message,
                    "action": str(v.action),
                }
                for v in self.violations
            ],
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
        }
