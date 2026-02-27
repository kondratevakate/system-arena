"""Decision point extraction from event logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from arena.core.event import Event, EventLog, EventType
from arena.core.snapshot import CaseSnapshot
from arena.core.types import Action, CaseId


@dataclass
class DecisionPointRule:
    """
    A rule that defines when a decision point occurs.

    Decision points are extracted by matching events against rules.
    """

    name: str
    trigger: EventType | str  # EventType or string pattern
    condition: str  # Python expression to evaluate
    allowed_actions: list[str] = field(default_factory=list)  # Action types allowed at this point

    def matches_trigger(self, event: Event) -> bool:
        """Check if an event matches the trigger."""
        if isinstance(self.trigger, EventType):
            return event.event_type == self.trigger
        # String pattern match (e.g., "state.changed")
        return event.event_type.value == self.trigger

    def evaluate_condition(
        self,
        snapshot: CaseSnapshot,
        event: Event,
        safe_globals: dict[str, Any] | None = None,
    ) -> bool:
        """
        Evaluate the condition expression against the snapshot and event.

        Uses restricted Python expression evaluation.
        """
        if not self.condition or self.condition.strip() == "":
            return True

        # Build evaluation context
        context = _build_eval_context(snapshot, event)

        if safe_globals:
            context.update(safe_globals)

        try:
            result = eval(self.condition, {"__builtins__": {}}, context)
            return bool(result)
        except Exception as e:
            # Log warning, default to False for safety
            import warnings

            warnings.warn(f"Error evaluating condition '{self.condition}': {e}")
            return False


@dataclass
class DecisionPoint:
    """
    A moment where a policy could intervene.

    This is the primary unit of evaluation in the benchmark.
    """

    # Identity
    case_id: CaseId
    decision_id: str  # Unique ID for this decision point

    # Temporal
    timestamp: datetime
    rule_name: str  # Which rule triggered this decision point

    # Context (no future leakage)
    snapshot: CaseSnapshot  # Case state at this moment
    triggering_event: Event  # The event that created this decision point

    # Action space
    allowed_actions: list[Action]  # What the policy can do

    # Historical ground truth (if available)
    historical_action: Action | None = None  # What actually happened
    outcome_label: str | None = None  # What outcome eventually occurred

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id:
            # Generate ID from case_id and timestamp
            self.decision_id = f"{self.case_id}_{self.timestamp.isoformat()}"


@dataclass
class ExtractionResult:
    """Result of decision point extraction."""

    decision_points: list[DecisionPoint]
    cases_processed: int
    events_processed: int
    rules_matched: dict[str, int]  # Count per rule
    errors: list[str] = field(default_factory=list)


def extract_decision_points(
    event_log: EventLog,
    rules: list[DecisionPointRule],
    action_builder: Callable[[DecisionPointRule, CaseSnapshot], list[Action]] | None = None,
    state_builder: Callable[[list[Event]], dict[str, Any]] | None = None,
) -> ExtractionResult:
    """
    Extract decision points from an event log using the given rules.

    This is the central function of the benchmark framework.

    Args:
        event_log: The event log to process
        rules: Decision point rules from the workflow manifest
        action_builder: Optional function to build allowed actions
        state_builder: Optional function to build state from events

    Returns:
        ExtractionResult with all decision points found
    """
    decision_points: list[DecisionPoint] = []
    rules_matched: dict[str, int] = {rule.name: 0 for rule in rules}
    errors: list[str] = []

    # Sort events by time to ensure correct temporal ordering
    sorted_log = event_log.sorted_by_time()

    # Group events by case
    cases: dict[CaseId, list[Event]] = {}
    for event in sorted_log:
        if event.case_id not in cases:
            cases[event.case_id] = []
        cases[event.case_id].append(event)

    # Process each case
    for case_id, case_events in cases.items():
        # Process events in temporal order
        for i, event in enumerate(case_events):
            # Build snapshot at this point (events up to and including this one)
            events_until_now = case_events[: i + 1]

            try:
                snapshot = CaseSnapshot.from_event_log(
                    case_id=case_id,
                    event_log=EventLog(events_until_now),
                    snapshot_time=event.timestamp,
                )
            except Exception as e:
                errors.append(f"Error building snapshot for case {case_id} at {event.timestamp}: {e}")
                continue

            # Check each rule
            for rule in rules:
                if not rule.matches_trigger(event):
                    continue

                if not rule.evaluate_condition(snapshot, event):
                    continue

                # Rule matched! Create decision point
                rules_matched[rule.name] += 1

                # Build allowed actions
                if action_builder:
                    allowed_actions = action_builder(rule, snapshot)
                else:
                    allowed_actions = _default_action_builder(rule)

                dp = DecisionPoint(
                    case_id=case_id,
                    decision_id=f"{case_id}_{event.timestamp.isoformat()}_{rule.name}",
                    timestamp=event.timestamp,
                    rule_name=rule.name,
                    snapshot=snapshot,
                    triggering_event=event,
                    allowed_actions=allowed_actions,
                )
                decision_points.append(dp)

    return ExtractionResult(
        decision_points=decision_points,
        cases_processed=len(cases),
        events_processed=len(sorted_log),
        rules_matched=rules_matched,
        errors=errors,
    )


def _build_eval_context(snapshot: CaseSnapshot, event: Event) -> dict[str, Any]:
    """Build the context for evaluating condition expressions."""
    context: dict[str, Any] = {}

    # State fields (e.g., state.status)
    context["state"] = type("State", (), snapshot.state)()

    # Event fields (e.g., event.payload.timer_name)
    context["event"] = type("EventContext", (), {
        "type": event.event_type.value,
        "payload": type("Payload", (), event.payload)(),
        **event.payload,
    })()

    # Computed features from snapshot
    context["days_since_last_contact"] = snapshot.days_since_last_interaction
    context["days_since_last_interaction"] = snapshot.days_since_last_interaction
    context["total_interactions"] = snapshot.total_interactions
    context["total_events"] = snapshot.total_events
    context["time_in_current_state_days"] = (
        snapshot.time_in_current_state.total_seconds() / 86400
    )

    # Feature dict
    for key, value in snapshot.features.items():
        context[key] = value

    # Also expose state fields directly for convenience
    for key, value in snapshot.state.items():
        if key not in context:  # Don't override existing
            context[key] = value

    # Common Python builtins (safe subset)
    context["True"] = True
    context["False"] = False
    context["None"] = None
    context["len"] = len
    context["min"] = min
    context["max"] = max
    context["abs"] = abs
    context["sum"] = sum
    context["any"] = any
    context["all"] = all

    return context


def _default_action_builder(rule: DecisionPointRule) -> list[Action]:
    """Build default allowed actions from rule definition."""
    if not rule.allowed_actions:
        # Default action space
        return [
            Action(action_type="wait"),
            Action(action_type="escalate"),
        ]

    return [Action(action_type=action_type) for action_type in rule.allowed_actions]
