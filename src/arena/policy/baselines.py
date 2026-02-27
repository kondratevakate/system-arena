"""Baseline policy implementations for benchmarking."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from arena.core.snapshot import CaseSnapshot
from arena.core.types import Action
from arena.policy.base import BasePolicy
from arena.policy.result import PolicyResult


class RandomPolicy(BasePolicy):
    """
    A policy that randomly selects from allowed actions.

    Useful as a lower-bound baseline.
    """

    def __init__(
        self,
        seed: int | None = None,
        name: str = "random_policy",
    ) -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        if not allowed_actions:
            raise ValueError("No allowed actions to choose from")

        action = self._rng.choice(allowed_actions)

        return PolicyResult(
            action=action,
            confidence=1.0 / len(allowed_actions),
            reasoning="Randomly selected from allowed actions",
        )


class ConstantPolicy(BasePolicy):
    """
    A policy that always returns the same action type.

    Useful for testing specific behaviors.
    """

    def __init__(
        self,
        action_type: str,
        params: dict[str, Any] | None = None,
        name: str = "constant_policy",
    ) -> None:
        super().__init__(name=name)
        self.action_type = action_type
        self.params = params or {}

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        # Find matching action in allowed actions
        matching = [
            a for a in allowed_actions if a.action_type == self.action_type
        ]

        if matching:
            return PolicyResult(
                action=matching[0],
                confidence=1.0,
                reasoning=f"Constant policy: always choose {self.action_type}",
            )

        # Action not in allowed set, choose first allowed as fallback
        if allowed_actions:
            return PolicyResult(
                action=allowed_actions[0],
                confidence=0.5,
                reasoning=f"Fallback: {self.action_type} not allowed",
                metadata={"intended_action": self.action_type},
            )

        raise ValueError("No allowed actions")


@dataclass
class Rule:
    """A single rule in a rule-based policy."""

    condition: str  # Python expression
    action_type: str
    action_params: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    name: str = ""


class RuleBasedPolicy(BasePolicy):
    """
    A policy that uses explicit if/then rules.

    Rules are evaluated in priority order (highest first).
    The first matching rule determines the action.
    """

    def __init__(
        self,
        rules: list[Rule],
        default_action: str = "wait",
        name: str = "rule_based_policy",
    ) -> None:
        super().__init__(name=name)
        self.rules = sorted(rules, key=lambda r: -r.priority)
        self.default_action = default_action

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        # Build evaluation context from snapshot
        context = self._build_context(snapshot)

        for rule in self.rules:
            if self._evaluate_rule(rule, context):
                # Find matching action
                action = self._find_action(rule.action_type, allowed_actions)

                if action:
                    return PolicyResult(
                        action=action,
                        confidence=1.0,
                        reasoning=f"Rule matched: {rule.name or rule.condition}",
                        metadata={"matched_rule": rule.name},
                    )

        # No rule matched, use default
        default = self._find_action(self.default_action, allowed_actions)

        if default:
            return PolicyResult(
                action=default,
                confidence=0.5,
                reasoning="No rules matched, using default action",
            )

        # Default not available, use first allowed
        if allowed_actions:
            return PolicyResult(
                action=allowed_actions[0],
                confidence=0.1,
                reasoning="Fallback to first allowed action",
            )

        raise ValueError("No allowed actions")

    def _build_context(self, snapshot: CaseSnapshot) -> dict[str, Any]:
        """Build evaluation context for rule conditions."""
        context: dict[str, Any] = {
            "state": snapshot.state,
            "days_since_last_contact": snapshot.days_since_last_interaction,
            "total_interactions": snapshot.total_interactions,
            "total_events": snapshot.total_events,
            "True": True,
            "False": False,
            "None": None,
        }

        # Add state fields directly
        context.update(snapshot.state)

        # Add features
        context.update(snapshot.features)

        return context

    def _evaluate_rule(self, rule: Rule, context: dict[str, Any]) -> bool:
        """Evaluate a rule condition."""
        try:
            result = eval(rule.condition, {"__builtins__": {}}, context)
            return bool(result)
        except Exception:
            return False

    def _find_action(
        self, action_type: str, allowed_actions: list[Action]
    ) -> Action | None:
        """Find an action by type in the allowed list."""
        for action in allowed_actions:
            if action.action_type == action_type:
                return action
        return None


@dataclass
class ThresholdConfig:
    """Configuration for a threshold-based decision."""

    field: str  # Field to check
    threshold: float  # Threshold value
    action_above: str  # Action when above threshold
    action_below: str  # Action when below threshold


class ThresholdPolicy(BasePolicy):
    """
    A policy that uses score thresholds to make decisions.

    Supports multiple thresholds with priority ordering.
    """

    def __init__(
        self,
        thresholds: list[ThresholdConfig],
        default_action: str = "wait",
        score_fn: Callable[[CaseSnapshot], float] | None = None,
        name: str = "threshold_policy",
    ) -> None:
        super().__init__(name=name)
        self.thresholds = thresholds
        self.default_action = default_action
        self.score_fn = score_fn

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        # If a custom score function is provided, use it
        if self.score_fn:
            score = self.score_fn(snapshot)
            return self._decide_by_score(score, allowed_actions)

        # Otherwise, check each threshold config
        for config in self.thresholds:
            value = self._get_field_value(snapshot, config.field)

            if value is None:
                continue

            if value >= config.threshold:
                action = self._find_action(config.action_above, allowed_actions)
            else:
                action = self._find_action(config.action_below, allowed_actions)

            if action:
                return PolicyResult(
                    action=action,
                    confidence=0.8,
                    reasoning=f"Threshold on {config.field}: {value} vs {config.threshold}",
                    metadata={
                        "field": config.field,
                        "value": value,
                        "threshold": config.threshold,
                    },
                )

        # Use default
        default = self._find_action(self.default_action, allowed_actions)
        if default:
            return PolicyResult(
                action=default,
                confidence=0.5,
                reasoning="No threshold matched, using default",
            )

        if allowed_actions:
            return PolicyResult(
                action=allowed_actions[0],
                confidence=0.1,
                reasoning="Fallback to first allowed action",
            )

        raise ValueError("No allowed actions")

    def _decide_by_score(
        self,
        score: float,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        """Make a decision based on a single score value."""
        # Simple binary threshold on the first config
        if self.thresholds:
            config = self.thresholds[0]
            if score >= config.threshold:
                action_type = config.action_above
            else:
                action_type = config.action_below

            action = self._find_action(action_type, allowed_actions)
            if action:
                return PolicyResult(
                    action=action,
                    confidence=min(1.0, abs(score)),
                    reasoning=f"Score {score:.2f} vs threshold {config.threshold}",
                    metadata={"score": score},
                )

        # Fallback
        if allowed_actions:
            return PolicyResult(
                action=allowed_actions[0],
                confidence=0.1,
                reasoning="Score-based fallback",
            )

        raise ValueError("No allowed actions")

    def _get_field_value(self, snapshot: CaseSnapshot, field: str) -> float | None:
        """Get a numeric field value from the snapshot."""
        # Check common computed fields
        if field == "days_since_last_contact":
            return snapshot.days_since_last_interaction
        if field == "total_interactions":
            return float(snapshot.total_interactions)
        if field == "total_events":
            return float(snapshot.total_events)

        # Check state
        if field in snapshot.state:
            value = snapshot.state[field]
            if isinstance(value, (int, float)):
                return float(value)

        # Check features
        if field in snapshot.features:
            value = snapshot.features[field]
            if isinstance(value, (int, float)):
                return float(value)

        return None

    def _find_action(
        self, action_type: str, allowed_actions: list[Action]
    ) -> Action | None:
        """Find an action by type."""
        for action in allowed_actions:
            if action.action_type == action_type:
                return action
        return None


# Convenience constructors
def create_followup_policy(
    days_threshold: float = 3.0,
    name: str = "followup_policy",
) -> RuleBasedPolicy:
    """Create a simple follow-up policy for sales workflows."""
    rules = [
        Rule(
            name="urgent_followup",
            condition=f"days_since_last_contact is not None and days_since_last_contact > {days_threshold * 2}",
            action_type="escalate",
            priority=10,
        ),
        Rule(
            name="standard_followup",
            condition=f"days_since_last_contact is not None and days_since_last_contact > {days_threshold}",
            action_type="send_reminder",
            priority=5,
        ),
        Rule(
            name="recent_contact",
            condition="days_since_last_contact is not None and days_since_last_contact <= 1",
            action_type="wait",
            priority=1,
        ),
    ]
    return RuleBasedPolicy(rules=rules, default_action="wait", name=name)
