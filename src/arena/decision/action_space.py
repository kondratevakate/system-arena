"""Allowed action space construction for decision points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from arena.core.snapshot import CaseSnapshot
from arena.core.types import Action


@dataclass
class ActionDefinition:
    """
    Definition of an action type that can appear in the action space.

    Actions are defined in the workflow manifest and instantiated
    with specific parameters at each decision point.
    """

    action_type: str
    params: list[str] = field(default_factory=list)  # Parameter names
    param_types: dict[str, str] = field(default_factory=dict)  # Optional type hints
    description: str = ""
    constraints: list[str] = field(default_factory=list)  # Conditions when action is valid

    def create_action(self, **kwargs: Any) -> Action:
        """Create an Action instance with the given parameters."""
        return Action(action_type=self.action_type, params=kwargs)

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate that the params match the definition. Returns error messages."""
        errors = []
        for param_name in self.params:
            if param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
        return errors


@dataclass
class ActionSpace:
    """
    The set of allowed actions at a decision point.

    The action space can be:
    - Static (same for all decision points)
    - Dynamic (varies based on case state)
    """

    actions: list[Action]
    action_definitions: list[ActionDefinition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __contains__(self, action: Action) -> bool:
        return any(
            a.action_type == action.action_type
            for a in self.actions
        )

    def get_action_types(self) -> list[str]:
        """Get the list of available action types."""
        return [a.action_type for a in self.actions]

    def filter_by_type(self, action_types: Sequence[str]) -> ActionSpace:
        """Filter to specific action types."""
        filtered = [a for a in self.actions if a.action_type in action_types]
        return ActionSpace(
            actions=filtered,
            action_definitions=[
                d for d in self.action_definitions if d.action_type in action_types
            ],
        )


class ActionSpaceBuilder:
    """
    Builds the action space for a decision point.

    Can apply dynamic filtering based on case state.
    """

    def __init__(self, definitions: list[ActionDefinition]) -> None:
        self.definitions = definitions
        self._filters: list[Callable[[Action, CaseSnapshot], bool]] = []

    def add_filter(
        self, filter_fn: Callable[[Action, CaseSnapshot], bool]
    ) -> ActionSpaceBuilder:
        """Add a filter that determines if an action is allowed."""
        self._filters.append(filter_fn)
        return self

    def build(
        self,
        snapshot: CaseSnapshot,
        param_generator: Callable[[ActionDefinition, CaseSnapshot], list[dict[str, Any]]] | None = None,
    ) -> ActionSpace:
        """
        Build the action space for a specific decision point.

        Args:
            snapshot: The case snapshot at the decision point
            param_generator: Optional function to generate parameter combinations
        """
        actions = []

        for defn in self.definitions:
            # Generate parameter combinations
            if param_generator:
                param_combos = param_generator(defn, snapshot)
            else:
                # Default: single action with no params, or params from snapshot
                param_combos = [{}]

            for params in param_combos:
                action = defn.create_action(**params)

                # Apply filters
                if all(f(action, snapshot) for f in self._filters):
                    actions.append(action)

        return ActionSpace(actions=actions, action_definitions=self.definitions)


def build_standard_action_space(
    action_types: list[str],
    snapshot: CaseSnapshot | None = None,
) -> ActionSpace:
    """
    Build a standard action space from action type names.

    This is a convenience function for simple workflows.
    """
    actions = [Action(action_type=at) for at in action_types]
    return ActionSpace(actions=actions)


# Common action spaces
BASIC_ACTION_SPACE = ActionSpace(
    actions=[
        Action(action_type="wait"),
        Action(action_type="contact"),
        Action(action_type="escalate"),
        Action(action_type="close"),
    ]
)

SALES_ACTION_SPACE = ActionSpace(
    actions=[
        Action(action_type="wait"),
        Action(action_type="send_email"),
        Action(action_type="send_sms"),
        Action(action_type="call"),
        Action(action_type="escalate"),
        Action(action_type="close", params={"reason": "no_response"}),
        Action(action_type="close", params={"reason": "qualified_out"}),
    ]
)
