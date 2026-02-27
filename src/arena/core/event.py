"""Canonical event schema for case-based workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator, Sequence
from uuid import UUID, uuid4

from arena.core.types import Actor, ActorType, CaseId, Channel, ObjectRef


class EventType(str, Enum):
    """
    Canonical low-level event types (domain-agnostic).

    Domain-specific semantics go in the event payload.
    """

    # Lifecycle
    ENTITY_CREATED = "entity.created"
    ENTITY_CLOSED = "entity.closed"
    ENTITY_REOPENED = "entity.reopened"

    # State
    STATE_CHANGED = "state.changed"

    # Interactions
    INTERACTION_OCCURRED = "interaction.occurred"

    # Decisions
    DECISION_REQUESTED = "decision.requested"
    DECISION_EXECUTED = "decision.executed"

    # Outcomes
    OUTCOME_RECORDED = "outcome.recorded"

    # Timers & escalation
    TIMER_ELAPSED = "timer.elapsed"
    ESCALATION_TRIGGERED = "escalation.triggered"

    # Assignment
    ASSIGNMENT_CHANGED = "assignment.changed"


@dataclass(frozen=True)
class Event:
    """
    Immutable event record in the event log.

    This is the atomic unit of the event sourcing system.
    All case state is derived from the sequence of events.
    """

    # Identity
    case_id: CaseId
    event_id: UUID = field(default_factory=uuid4)

    # Temporal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence_number: int = 0  # Monotonic within case

    # Classification
    event_type: EventType = EventType.INTERACTION_OCCURRED
    workflow_type: str = ""  # e.g., "sales_lead", "screening_episode"

    # Actor & Channel
    actor: Actor = field(
        default_factory=lambda: Actor(actor_type=ActorType.SYSTEM, actor_id="system")
    )
    channel: Channel | None = None

    # Payload (domain-specific content)
    payload: dict[str, Any] = field(default_factory=dict)

    # Optional state snapshots (debug artifact, not mandatory)
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None

    # Labels (for evaluation)
    outcome_labels: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Provenance
    source_system: str | None = None
    correlation_id: UUID | None = None

    # Object-centric process mining: related objects beyond the primary case_id
    # This enables multi-entity process analysis without losing structural relationships
    # Example: a quote_sent event might reference: lead, clinic, patient, assistant
    object_refs: tuple[ObjectRef, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "event_id": str(self.event_id),
            "case_id": str(self.case_id),
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "event_type": self.event_type.value,
            "workflow_type": self.workflow_type,
            "actor": {
                "actor_type": self.actor.actor_type.value,
                "actor_id": self.actor.actor_id,
                "name": self.actor.name,
                "metadata": self.actor.metadata,
            },
            "channel": self.channel.value if self.channel else None,
            "payload": self.payload,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "outcome_labels": self.outcome_labels,
            "metadata": self.metadata,
            "source_system": self.source_system,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Deserialize from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            case_id=CaseId(UUID(data["case_id"])),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sequence_number=data.get("sequence_number", 0),
            event_type=EventType(data["event_type"]),
            workflow_type=data.get("workflow_type", ""),
            actor=Actor(
                actor_type=ActorType(data["actor"]["actor_type"]),
                actor_id=data["actor"]["actor_id"],
                name=data["actor"].get("name"),
                metadata=data["actor"].get("metadata", {}),
            ),
            channel=Channel(data["channel"]) if data.get("channel") else None,
            payload=data.get("payload", {}),
            state_before=data.get("state_before"),
            state_after=data.get("state_after"),
            outcome_labels=data.get("outcome_labels", {}),
            metadata=data.get("metadata", {}),
            source_system=data.get("source_system"),
            correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") else None,
        )


@dataclass
class EventLog:
    """
    Collection of events with query capabilities.

    Provides efficient access patterns for replay and analysis.
    """

    events: list[Event] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __getitem__(self, idx: int) -> Event:
        return self.events[idx]

    @property
    def case_ids(self) -> set[CaseId]:
        """Get all unique case IDs in the log."""
        return {e.case_id for e in self.events}

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        """Get the time range covered by events."""
        if not self.events:
            return None
        timestamps = [e.timestamp for e in self.events]
        return min(timestamps), max(timestamps)

    def filter_by_case(self, case_id: CaseId) -> EventLog:
        """Get all events for a specific case."""
        return EventLog([e for e in self.events if e.case_id == case_id])

    def filter_by_time_range(self, start: datetime, end: datetime) -> EventLog:
        """Get events within a time window (inclusive)."""
        return EventLog([e for e in self.events if start <= e.timestamp <= end])

    def filter_by_type(self, event_types: Sequence[EventType]) -> EventLog:
        """Filter to specific event types."""
        type_set = set(event_types)
        return EventLog([e for e in self.events if e.event_type in type_set])

    def filter_until(self, timestamp: datetime) -> EventLog:
        """Get events up to (and including) a timestamp. Critical for no-future-leakage."""
        return EventLog([e for e in self.events if e.timestamp <= timestamp])

    def filter_by_object(self, object_type: str, object_id: str | None = None) -> EventLog:
        """
        Filter events by related object (object-centric process mining).

        Args:
            object_type: Type of object to filter by (e.g., "clinic", "patient")
            object_id: Optional specific object ID

        Returns:
            EventLog with events related to the specified object
        """
        def matches(event: Event) -> bool:
            for ref in event.object_refs:
                if ref.object_type == object_type:
                    if object_id is None or ref.object_id == object_id:
                        return True
            return False

        return EventLog([e for e in self.events if matches(e)])

    @property
    def object_types(self) -> set[str]:
        """Get all unique object types referenced in the log."""
        types: set[str] = set()
        for event in self.events:
            for ref in event.object_refs:
                types.add(ref.object_type)
        return types

    def get_decision_points(self) -> list[Event]:
        """Extract events where decisions were requested."""
        return [e for e in self.events if e.event_type == EventType.DECISION_REQUESTED]

    def sorted_by_time(self) -> EventLog:
        """Return a new EventLog sorted by timestamp."""
        return EventLog(sorted(self.events, key=lambda e: (e.timestamp, e.sequence_number)))

    def append(self, event: Event) -> None:
        """Add an event to the log."""
        self.events.append(event)

    def extend(self, events: Sequence[Event]) -> None:
        """Add multiple events to the log."""
        self.events.extend(events)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all events to dictionaries."""
        return [e.to_dict() for e in self.events]

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> EventLog:
        """Deserialize from list of dictionaries."""
        return cls([Event.from_dict(d) for d in data])
