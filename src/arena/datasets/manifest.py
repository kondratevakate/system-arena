"""Workflow manifest schema and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from arena.core.event import EventType
from arena.decision.extractor import DecisionPointRule
from arena.decision.labeler import OutcomeCategory, OutcomeDefinition
from arena.replay.runner import Constraint


@dataclass
class CaseConfig:
    """Configuration for case identification in the dataset."""

    id_field: str = "case_id"
    timestamp_field: str = "timestamp"


@dataclass
class ActionConfig:
    """Configuration for an allowed action."""

    action: str
    params: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ConstraintConfig:
    """Configuration for a constraint."""

    name: str
    rule: str
    severity: str = "violation"
    message: str = ""


@dataclass
class OutcomeConfig:
    """Configuration for outcomes."""

    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)


@dataclass
class DecisionPointConfig:
    """Configuration for a decision point rule."""

    name: str
    trigger: str
    condition: str = ""
    allowed_actions: list[str] = field(default_factory=list)


@dataclass
class WorkflowManifest:
    """
    Complete workflow manifest loaded from YAML.

    Defines:
    - How to identify cases in the data
    - When decision points occur
    - What actions are allowed
    - What constraints must be respected
    - How to label outcomes
    """

    name: str
    version: str = "1.0"
    description: str = ""

    # Case configuration
    case: CaseConfig = field(default_factory=CaseConfig)

    # Decision points
    decision_points: list[DecisionPointConfig] = field(default_factory=list)

    # Allowed actions
    allowed_actions: list[ActionConfig] = field(default_factory=list)

    # Constraints
    constraints: list[ConstraintConfig] = field(default_factory=list)

    # Outcomes
    outcomes: OutcomeConfig = field(default_factory=OutcomeConfig)

    # Optional Python hooks module
    hooks_module: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_decision_rules(self) -> list[DecisionPointRule]:
        """Convert decision point configs to DecisionPointRule objects."""
        rules = []

        for config in self.decision_points:
            # Parse trigger to EventType if possible
            trigger: EventType | str
            try:
                trigger = EventType(config.trigger)
            except ValueError:
                trigger = config.trigger

            rules.append(
                DecisionPointRule(
                    name=config.name,
                    trigger=trigger,
                    condition=config.condition,
                    allowed_actions=config.allowed_actions or self._get_all_action_types(),
                )
            )

        return rules

    def get_constraints(self) -> list[Constraint]:
        """Convert constraint configs to Constraint objects."""
        return [
            Constraint(
                name=c.name,
                rule=c.rule,
                severity=c.severity,
                message=c.message,
            )
            for c in self.constraints
        ]

    def get_outcome_definitions(self) -> list[OutcomeDefinition]:
        """Convert outcome configs to OutcomeDefinition objects."""
        definitions = []

        for outcome in self.outcomes.positive:
            definitions.append(
                OutcomeDefinition(
                    name=outcome,
                    category=OutcomeCategory.POSITIVE,
                    detection_rule=f"payload.get('outcome') == '{outcome}'",
                    priority=10,
                )
            )

        for outcome in self.outcomes.negative:
            definitions.append(
                OutcomeDefinition(
                    name=outcome,
                    category=OutcomeCategory.NEGATIVE,
                    detection_rule=f"payload.get('outcome') == '{outcome}'",
                    priority=10,
                )
            )

        for outcome in self.outcomes.neutral:
            definitions.append(
                OutcomeDefinition(
                    name=outcome,
                    category=OutcomeCategory.NEUTRAL,
                    detection_rule=f"payload.get('outcome') == '{outcome}'",
                    priority=5,
                )
            )

        return definitions

    def _get_all_action_types(self) -> list[str]:
        """Get all defined action types."""
        return [a.action for a in self.allowed_actions]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowManifest:
        """Create a manifest from a dictionary."""
        # Parse case config
        case_data = data.get("case", {})
        case_config = CaseConfig(
            id_field=case_data.get("id_field", "case_id"),
            timestamp_field=case_data.get("timestamp_field", "timestamp"),
        )

        # Parse decision points
        decision_points = []
        for dp in data.get("decision_points", []):
            decision_points.append(
                DecisionPointConfig(
                    name=dp.get("name", ""),
                    trigger=dp.get("trigger", "state.changed"),
                    condition=dp.get("condition", ""),
                    allowed_actions=dp.get("allowed_actions", []),
                )
            )

        # Parse allowed actions
        allowed_actions = []
        for action in data.get("allowed_actions", []):
            if isinstance(action, str):
                allowed_actions.append(ActionConfig(action=action))
            else:
                allowed_actions.append(
                    ActionConfig(
                        action=action.get("action", ""),
                        params=action.get("params", []),
                        description=action.get("description", ""),
                    )
                )

        # Parse constraints
        constraints = []
        for c in data.get("constraints", []):
            constraints.append(
                ConstraintConfig(
                    name=c.get("name", ""),
                    rule=c.get("rule", ""),
                    severity=c.get("severity", "violation"),
                    message=c.get("message", ""),
                )
            )

        # Parse outcomes
        outcomes_data = data.get("outcomes", {})
        outcomes = OutcomeConfig(
            positive=outcomes_data.get("positive", []),
            negative=outcomes_data.get("negative", []),
            neutral=outcomes_data.get("neutral", []),
        )

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            case=case_config,
            decision_points=decision_points,
            allowed_actions=allowed_actions,
            constraints=constraints,
            outcomes=outcomes,
            hooks_module=data.get("hooks_module"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> WorkflowManifest:
        """Parse a manifest from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "case": {
                "id_field": self.case.id_field,
                "timestamp_field": self.case.timestamp_field,
            },
            "decision_points": [
                {
                    "name": dp.name,
                    "trigger": dp.trigger,
                    "condition": dp.condition,
                    "allowed_actions": dp.allowed_actions,
                }
                for dp in self.decision_points
            ],
            "allowed_actions": [
                {
                    "action": a.action,
                    "params": a.params,
                    "description": a.description,
                }
                for a in self.allowed_actions
            ],
            "constraints": [
                {
                    "name": c.name,
                    "rule": c.rule,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.constraints
            ],
            "outcomes": {
                "positive": self.outcomes.positive,
                "negative": self.outcomes.negative,
                "neutral": self.outcomes.neutral,
            },
            "hooks_module": self.hooks_module,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        """Convert manifest to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


def load_manifest(path: str | Path) -> WorkflowManifest:
    """
    Load a workflow manifest from a YAML file.

    Args:
        path: Path to the manifest YAML file

    Returns:
        Parsed WorkflowManifest
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path) as f:
        content = f.read()

    return WorkflowManifest.from_yaml(content)
