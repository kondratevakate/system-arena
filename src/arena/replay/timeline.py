"""Timeline management for replay with no future leakage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator

from arena.core.event import Event, EventLog
from arena.core.types import CaseId
from arena.decision.extractor import DecisionPoint


@dataclass
class TimelineEntry:
    """A single entry in the replay timeline."""

    timestamp: datetime
    entry_type: str  # "decision_point" or "event"
    case_id: CaseId
    data: DecisionPoint | Event


@dataclass
class Timeline:
    """
    Manages the temporal ordering of events and decision points.

    Critical for replay: ensures no future leakage by enforcing
    strict temporal ordering and providing only past information.
    """

    entries: list[TimelineEntry] = field(default_factory=list)
    _current_index: int = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TimelineEntry]:
        return iter(self.entries)

    @classmethod
    def from_decision_points(
        cls,
        decision_points: list[DecisionPoint],
    ) -> Timeline:
        """Create a timeline from decision points only."""
        entries = [
            TimelineEntry(
                timestamp=dp.timestamp,
                entry_type="decision_point",
                case_id=dp.case_id,
                data=dp,
            )
            for dp in decision_points
        ]

        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)

        return cls(entries=entries)

    @classmethod
    def from_events_and_decision_points(
        cls,
        event_log: EventLog,
        decision_points: list[DecisionPoint],
    ) -> Timeline:
        """
        Create a timeline interleaving events and decision points.

        Useful for understanding the full context during replay.
        """
        entries = []

        # Add events
        for event in event_log:
            entries.append(
                TimelineEntry(
                    timestamp=event.timestamp,
                    entry_type="event",
                    case_id=event.case_id,
                    data=event,
                )
            )

        # Add decision points
        for dp in decision_points:
            entries.append(
                TimelineEntry(
                    timestamp=dp.timestamp,
                    entry_type="decision_point",
                    case_id=dp.case_id,
                    data=dp,
                )
            )

        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)

        return cls(entries=entries)

    def decision_points_only(self) -> list[DecisionPoint]:
        """Get only the decision point entries."""
        return [
            e.data
            for e in self.entries
            if e.entry_type == "decision_point" and isinstance(e.data, DecisionPoint)
        ]

    def filter_by_case(self, case_id: CaseId) -> Timeline:
        """Get timeline entries for a specific case."""
        filtered = [e for e in self.entries if e.case_id == case_id]
        return Timeline(entries=filtered)

    def filter_by_time_range(
        self, start: datetime, end: datetime
    ) -> Timeline:
        """Get entries within a time range."""
        filtered = [
            e for e in self.entries if start <= e.timestamp <= end
        ]
        return Timeline(entries=filtered)

    def get_entries_before(self, timestamp: datetime) -> Timeline:
        """
        Get all entries strictly before a timestamp.

        This is the key method for no-future-leakage.
        """
        filtered = [e for e in self.entries if e.timestamp < timestamp]
        return Timeline(entries=filtered)

    def get_entries_until(self, timestamp: datetime) -> Timeline:
        """Get all entries up to and including a timestamp."""
        filtered = [e for e in self.entries if e.timestamp <= timestamp]
        return Timeline(entries=filtered)

    def reset(self) -> None:
        """Reset the timeline to the beginning."""
        self._current_index = 0

    def advance(self) -> TimelineEntry | None:
        """
        Advance to the next entry in the timeline.

        Returns None when the timeline is exhausted.
        """
        if self._current_index >= len(self.entries):
            return None

        entry = self.entries[self._current_index]
        self._current_index += 1
        return entry

    def peek(self) -> TimelineEntry | None:
        """Look at the next entry without advancing."""
        if self._current_index >= len(self.entries):
            return None
        return self.entries[self._current_index]

    @property
    def current_time(self) -> datetime | None:
        """Get the timestamp of the current position."""
        if self._current_index == 0:
            return self.entries[0].timestamp if self.entries else None
        if self._current_index > len(self.entries):
            return None
        return self.entries[self._current_index - 1].timestamp

    @property
    def is_exhausted(self) -> bool:
        """Check if the timeline has been fully processed."""
        return self._current_index >= len(self.entries)

    @property
    def time_span(self) -> timedelta | None:
        """Get the total time span covered by the timeline."""
        if not self.entries:
            return None
        return self.entries[-1].timestamp - self.entries[0].timestamp


class ReplayContext:
    """
    Context for replay that enforces no-future-leakage.

    Wraps a timeline and ensures that only past information
    is accessible at each decision point.
    """

    def __init__(self, timeline: Timeline, event_log: EventLog) -> None:
        self.timeline = timeline
        self.event_log = event_log
        self._events_by_case: dict[CaseId, list[Event]] = {}

        # Index events by case for efficient lookup
        for event in event_log.sorted_by_time():
            if event.case_id not in self._events_by_case:
                self._events_by_case[event.case_id] = []
            self._events_by_case[event.case_id].append(event)

    def get_visible_events(
        self,
        case_id: CaseId,
        at_time: datetime,
    ) -> list[Event]:
        """
        Get events visible at a specific point in time.

        Only returns events that occurred BEFORE at_time.
        This is the core of no-future-leakage enforcement.
        """
        case_events = self._events_by_case.get(case_id, [])
        return [e for e in case_events if e.timestamp < at_time]

    def get_all_visible_events(self, at_time: datetime) -> EventLog:
        """Get all events across all cases visible at a point in time."""
        return self.event_log.filter_until(at_time)

    def validate_no_future_leakage(
        self,
        decision_point: DecisionPoint,
    ) -> list[str]:
        """
        Validate that a decision point snapshot contains no future information.

        Returns a list of warnings if any future leakage is detected.
        """
        warnings = []

        # Check that snapshot time matches decision point time
        if decision_point.snapshot.snapshot_time > decision_point.timestamp:
            warnings.append(
                f"Snapshot time {decision_point.snapshot.snapshot_time} is after "
                f"decision point time {decision_point.timestamp}"
            )

        # Check that all events in snapshot history are before decision time
        for event in decision_point.snapshot.event_history:
            if event.timestamp > decision_point.timestamp:
                warnings.append(
                    f"Event {event.event_id} at {event.timestamp} is after "
                    f"decision point time {decision_point.timestamp}"
                )

        return warnings
