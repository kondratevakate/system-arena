"""Outcome labeling for decision points."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Sequence

from arena.core.event import Event, EventLog, EventType
from arena.core.types import CaseId
from arena.decision.extractor import DecisionPoint


class OutcomeCategory(str, Enum):
    """Categories for outcome labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class OutcomeDefinition:
    """Definition of an outcome type."""

    name: str
    category: OutcomeCategory
    detection_rule: str  # Python expression to detect this outcome
    priority: int = 0  # Higher priority outcomes take precedence


@dataclass
class OutcomeLabel:
    """A labeled outcome for a decision point."""

    label: str
    category: OutcomeCategory
    detected_at: datetime
    triggering_event: Event | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelingResult:
    """Result of labeling decision points."""

    labeled_points: list[DecisionPoint]
    labels_by_category: dict[OutcomeCategory, int]
    unlabeled_count: int
    errors: list[str] = field(default_factory=list)


class OutcomeLabeler:
    """
    Labels decision points with their eventual outcomes.

    This is used to:
    1. Provide ground truth for evaluation
    2. Label historical actions for replay
    3. Compute outcome-based metrics
    """

    def __init__(
        self,
        outcome_definitions: list[OutcomeDefinition],
        lookback_window: timedelta | None = None,
        lookahead_window: timedelta | None = None,
    ) -> None:
        """
        Initialize the labeler.

        Args:
            outcome_definitions: Definitions of possible outcomes
            lookback_window: How far back to look for relevant events
            lookahead_window: How far ahead to look for outcomes
                              (None = until end of case)
        """
        self.outcome_definitions = sorted(
            outcome_definitions, key=lambda x: -x.priority
        )
        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window

    def label_decision_points(
        self,
        decision_points: Sequence[DecisionPoint],
        event_log: EventLog,
    ) -> LabelingResult:
        """
        Label decision points with their outcomes.

        For each decision point, looks forward in time to find
        outcome events and applies labeling rules.
        """
        labeled_points: list[DecisionPoint] = []
        labels_by_category: dict[OutcomeCategory, int] = {
            cat: 0 for cat in OutcomeCategory
        }
        unlabeled_count = 0
        errors: list[str] = []

        # Group events by case
        events_by_case: dict[CaseId, list[Event]] = {}
        for event in event_log.sorted_by_time():
            if event.case_id not in events_by_case:
                events_by_case[event.case_id] = []
            events_by_case[event.case_id].append(event)

        for dp in decision_points:
            case_events = events_by_case.get(dp.case_id, [])

            # Find future events (after decision point)
            future_events = [
                e for e in case_events if e.timestamp > dp.timestamp
            ]

            # Apply lookahead window if set
            if self.lookahead_window:
                cutoff = dp.timestamp + self.lookahead_window
                future_events = [e for e in future_events if e.timestamp <= cutoff]

            # Try to find an outcome
            outcome = self._find_outcome(dp, future_events)

            if outcome:
                # Create a new decision point with the label
                labeled_dp = DecisionPoint(
                    case_id=dp.case_id,
                    decision_id=dp.decision_id,
                    timestamp=dp.timestamp,
                    rule_name=dp.rule_name,
                    snapshot=dp.snapshot,
                    triggering_event=dp.triggering_event,
                    allowed_actions=dp.allowed_actions,
                    historical_action=dp.historical_action,
                    outcome_label=outcome.label,
                    metadata={**dp.metadata, "outcome": outcome.__dict__},
                )
                labeled_points.append(labeled_dp)
                labels_by_category[outcome.category] += 1
            else:
                # No outcome found, keep original
                labeled_points.append(dp)
                unlabeled_count += 1
                labels_by_category[OutcomeCategory.UNKNOWN] += 1

        return LabelingResult(
            labeled_points=labeled_points,
            labels_by_category=labels_by_category,
            unlabeled_count=unlabeled_count,
            errors=errors,
        )

    def _find_outcome(
        self,
        dp: DecisionPoint,
        future_events: list[Event],
    ) -> OutcomeLabel | None:
        """
        Find the outcome for a decision point by checking future events.

        Returns the highest-priority matching outcome.
        """
        for outcome_def in self.outcome_definitions:
            for event in future_events:
                if self._matches_outcome_rule(outcome_def, event, dp):
                    return OutcomeLabel(
                        label=outcome_def.name,
                        category=outcome_def.category,
                        detected_at=event.timestamp,
                        triggering_event=event,
                    )
        return None

    def _matches_outcome_rule(
        self,
        outcome_def: OutcomeDefinition,
        event: Event,
        dp: DecisionPoint,
    ) -> bool:
        """Check if an event matches an outcome detection rule."""
        if not outcome_def.detection_rule:
            return False

        context = {
            "event": event,
            "event_type": event.event_type.value,
            "payload": event.payload,
            "outcome_labels": event.outcome_labels,
            "dp": dp,
            "True": True,
            "False": False,
            "None": None,
        }

        # Add payload fields directly
        context.update(event.payload)
        context.update(event.outcome_labels)

        try:
            result = eval(outcome_def.detection_rule, {"__builtins__": {}}, context)
            return bool(result)
        except Exception:
            return False


def label_from_event_outcomes(
    decision_points: Sequence[DecisionPoint],
    event_log: EventLog,
) -> list[DecisionPoint]:
    """
    Simple labeling using outcome_labels already present in events.

    This is useful when the event log already has labeled outcomes.
    """
    labeled = []
    events_by_case = {}

    for event in event_log.sorted_by_time():
        if event.case_id not in events_by_case:
            events_by_case[event.case_id] = []
        events_by_case[event.case_id].append(event)

    for dp in decision_points:
        case_events = events_by_case.get(dp.case_id, [])

        # Find the last OUTCOME_RECORDED event
        outcome_events = [
            e
            for e in case_events
            if e.event_type == EventType.OUTCOME_RECORDED
            and e.timestamp > dp.timestamp
        ]

        if outcome_events:
            last_outcome = outcome_events[-1]
            outcome_label = last_outcome.outcome_labels.get(
                "label", last_outcome.payload.get("outcome")
            )
            labeled.append(
                DecisionPoint(
                    case_id=dp.case_id,
                    decision_id=dp.decision_id,
                    timestamp=dp.timestamp,
                    rule_name=dp.rule_name,
                    snapshot=dp.snapshot,
                    triggering_event=dp.triggering_event,
                    allowed_actions=dp.allowed_actions,
                    historical_action=dp.historical_action,
                    outcome_label=outcome_label,
                    metadata=dp.metadata,
                )
            )
        else:
            labeled.append(dp)

    return labeled


# Convenience constructors for common outcome definitions
def sales_outcomes() -> list[OutcomeDefinition]:
    """Standard outcome definitions for sales workflows."""
    return [
        OutcomeDefinition(
            name="meeting_booked",
            category=OutcomeCategory.POSITIVE,
            detection_rule="payload.get('outcome') == 'meeting_booked' or 'meeting' in str(payload).lower()",
            priority=10,
        ),
        OutcomeDefinition(
            name="deal_closed_won",
            category=OutcomeCategory.POSITIVE,
            detection_rule="payload.get('outcome') == 'closed_won' or payload.get('status') == 'won'",
            priority=20,
        ),
        OutcomeDefinition(
            name="deal_closed_lost",
            category=OutcomeCategory.NEGATIVE,
            detection_rule="payload.get('outcome') == 'closed_lost' or payload.get('status') == 'lost'",
            priority=15,
        ),
        OutcomeDefinition(
            name="lead_went_cold",
            category=OutcomeCategory.NEGATIVE,
            detection_rule="payload.get('outcome') == 'cold' or payload.get('status') == 'cold'",
            priority=5,
        ),
        OutcomeDefinition(
            name="reply_received",
            category=OutcomeCategory.POSITIVE,
            detection_rule="event_type == 'interaction.occurred' and payload.get('direction') == 'inbound'",
            priority=3,
        ),
    ]
