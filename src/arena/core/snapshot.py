"""Point-in-time case snapshots for agent observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from arena.core.event import Event, EventLog, EventType
from arena.core.types import CaseId


@dataclass
class CaseSnapshot:
    """
    Point-in-time view of a case.

    Used by agents to observe the case state without future information.
    This is the primary input to the Policy.decide() method.

    Critical: Contains ONLY information available at snapshot_time.
    No future leakage is allowed.
    """

    # Identity
    case_id: CaseId
    workflow_type: str

    # Temporal
    snapshot_time: datetime
    case_created_at: datetime

    # Current state (derived from events up to snapshot_time)
    state: dict[str, Any] = field(default_factory=dict)

    # Event history (only events up to snapshot_time)
    event_history: list[Event] = field(default_factory=list)

    # Computed features (derived from history)
    time_since_created: timedelta = field(default_factory=lambda: timedelta(0))
    time_in_current_state: timedelta = field(default_factory=lambda: timedelta(0))
    total_interactions: int = 0
    total_events: int = 0
    last_interaction_time: datetime | None = None
    days_since_last_interaction: float | None = None

    # Additional computed features (workflow-specific)
    features: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event_log(
        cls,
        case_id: CaseId,
        event_log: EventLog,
        snapshot_time: datetime,
        state_builder: StateBuilder | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> CaseSnapshot:
        """
        Build a snapshot from an event log at a specific point in time.

        Args:
            case_id: The case to snapshot
            event_log: Full event log (will be filtered to case and time)
            snapshot_time: The point in time to snapshot (no future events)
            state_builder: Optional custom state builder
            feature_extractor: Optional custom feature extractor
        """
        # Filter to this case, up to snapshot_time (NO FUTURE LEAKAGE)
        case_events = (
            event_log.filter_by_case(case_id).filter_until(snapshot_time).sorted_by_time()
        )

        if not case_events.events:
            raise ValueError(f"No events found for case {case_id} before {snapshot_time}")

        events = case_events.events
        first_event = events[0]
        workflow_type = first_event.workflow_type

        # Build state from events
        if state_builder:
            state = state_builder.build_state(events)
        else:
            state = _default_build_state(events)

        # Compute standard features
        case_created_at = first_event.timestamp
        time_since_created = snapshot_time - case_created_at

        # Time in current state (since last STATE_CHANGED)
        state_changes = [e for e in events if e.event_type == EventType.STATE_CHANGED]
        if state_changes:
            time_in_current_state = snapshot_time - state_changes[-1].timestamp
        else:
            time_in_current_state = time_since_created

        # Interaction stats
        interaction_events = [
            e for e in events if e.event_type == EventType.INTERACTION_OCCURRED
        ]
        total_interactions = len(interaction_events)
        last_interaction_time = interaction_events[-1].timestamp if interaction_events else None
        days_since_last_interaction = (
            (snapshot_time - last_interaction_time).total_seconds() / 86400
            if last_interaction_time
            else None
        )

        # Extract additional features
        if feature_extractor:
            features = feature_extractor.extract(events, snapshot_time)
        else:
            features = {}

        return cls(
            case_id=case_id,
            workflow_type=workflow_type,
            snapshot_time=snapshot_time,
            case_created_at=case_created_at,
            state=state,
            event_history=events,
            time_since_created=time_since_created,
            time_in_current_state=time_in_current_state,
            total_interactions=total_interactions,
            total_events=len(events),
            last_interaction_time=last_interaction_time,
            days_since_last_interaction=days_since_last_interaction,
            features=features,
        )

    def get_recent_events(self, n: int = 10) -> list[Event]:
        """Get the N most recent events."""
        return self.event_history[-n:]

    def get_events_by_type(self, event_type: EventType) -> list[Event]:
        """Filter history by event type."""
        return [e for e in self.event_history if e.event_type == event_type]

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the current state."""
        return self.state.get(key, default)

    def get_feature(self, key: str, default: Any = None) -> Any:
        """Get a computed feature value."""
        return self.features.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes full event history for size)."""
        return {
            "case_id": str(self.case_id),
            "workflow_type": self.workflow_type,
            "snapshot_time": self.snapshot_time.isoformat(),
            "case_created_at": self.case_created_at.isoformat(),
            "state": self.state,
            "time_since_created_seconds": self.time_since_created.total_seconds(),
            "time_in_current_state_seconds": self.time_in_current_state.total_seconds(),
            "total_interactions": self.total_interactions,
            "total_events": self.total_events,
            "last_interaction_time": (
                self.last_interaction_time.isoformat() if self.last_interaction_time else None
            ),
            "days_since_last_interaction": self.days_since_last_interaction,
            "features": self.features,
            "event_count_in_history": len(self.event_history),
        }


class StateBuilder:
    """
    Protocol for building state from events.

    Implement this for workflow-specific state reconstruction.
    """

    def build_state(self, events: list[Event]) -> dict[str, Any]:
        """Build the current state from the event sequence."""
        raise NotImplementedError


class FeatureExtractor:
    """
    Protocol for extracting features from events.

    Implement this for workflow-specific feature engineering.
    """

    def extract(self, events: list[Event], snapshot_time: datetime) -> dict[str, Any]:
        """Extract features from the event sequence."""
        raise NotImplementedError


def _default_build_state(events: list[Event]) -> dict[str, Any]:
    """
    Default state builder: use the most recent state_after, or accumulate from payloads.
    """
    state: dict[str, Any] = {}

    for event in events:
        # If event has state_after, use it as the new state
        if event.state_after:
            state = dict(event.state_after)
        # Otherwise, merge relevant payload fields
        elif event.event_type == EventType.STATE_CHANGED and "new_state" in event.payload:
            state["status"] = event.payload["new_state"]
        elif event.event_type == EventType.ASSIGNMENT_CHANGED and "assignee" in event.payload:
            state["assignee"] = event.payload["assignee"]

    return state
