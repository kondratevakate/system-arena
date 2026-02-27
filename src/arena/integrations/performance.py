"""Performance analysis and time-in-state visualization.

This module provides tools for analyzing process performance:
- Time spent in each state
- Bottleneck detection
- Throughput analysis
- SLA compliance checking

Based on Apromore's performance mining capabilities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from arena.core.event import Event, EventLog, EventType
from arena.core.types import CaseId


@dataclass
class TimeInStateStats:
    """Statistics for time spent in a particular state."""

    state: str
    total_time: timedelta
    case_count: int
    entry_count: int  # How many times this state was entered

    # Distribution
    min_time: timedelta
    max_time: timedelta
    avg_time: timedelta
    median_time: timedelta | None = None
    p90_time: timedelta | None = None  # 90th percentile

    # Cases that spent the most time in this state
    longest_cases: list[tuple[CaseId, timedelta]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "total_time_seconds": self.total_time.total_seconds(),
            "case_count": self.case_count,
            "entry_count": self.entry_count,
            "min_time_seconds": self.min_time.total_seconds(),
            "max_time_seconds": self.max_time.total_seconds(),
            "avg_time_seconds": self.avg_time.total_seconds(),
            "median_time_seconds": (
                self.median_time.total_seconds() if self.median_time else None
            ),
            "p90_time_seconds": (
                self.p90_time.total_seconds() if self.p90_time else None
            ),
        }


@dataclass
class TransitionStats:
    """Statistics for a state transition."""

    from_state: str
    to_state: str
    count: int
    avg_time: timedelta
    min_time: timedelta
    max_time: timedelta

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "count": self.count,
            "avg_time_seconds": self.avg_time.total_seconds(),
            "min_time_seconds": self.min_time.total_seconds(),
            "max_time_seconds": self.max_time.total_seconds(),
        }


@dataclass
class BottleneckInfo:
    """Information about a process bottleneck."""

    state: str
    severity: str  # "high", "medium", "low"
    avg_wait_time: timedelta
    impact_score: float  # 0-1, how much this bottleneck impacts overall throughput
    affected_cases: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "severity": self.severity,
            "avg_wait_time_seconds": self.avg_wait_time.total_seconds(),
            "impact_score": self.impact_score,
            "affected_cases": self.affected_cases,
            "recommendation": self.recommendation,
        }


@dataclass
class PerformanceReport:
    """Complete performance analysis report."""

    # Overall metrics
    total_cases: int
    avg_case_duration: timedelta
    min_case_duration: timedelta
    max_case_duration: timedelta
    throughput_per_day: float

    # State-level analysis
    time_in_state: list[TimeInStateStats]
    transitions: list[TransitionStats]
    bottlenecks: list[BottleneckInfo]

    # Time range
    analysis_start: datetime
    analysis_end: datetime

    def summary(self) -> str:
        """Generate a text summary of the performance analysis."""
        lines = [
            "Performance Analysis Report",
            "=" * 60,
            "",
            f"Analysis Period: {self.analysis_start.date()} to {self.analysis_end.date()}",
            f"Total Cases: {self.total_cases}",
            f"Avg Case Duration: {_format_duration(self.avg_case_duration)}",
            f"Throughput: {self.throughput_per_day:.1f} cases/day",
            "",
            "Time in State (sorted by avg time):",
            "-" * 40,
        ]

        # Sort by avg time descending
        sorted_states = sorted(
            self.time_in_state,
            key=lambda s: s.avg_time.total_seconds(),
            reverse=True,
        )

        for stats in sorted_states[:10]:
            lines.append(
                f"  {stats.state}: avg={_format_duration(stats.avg_time)}, "
                f"entries={stats.entry_count}"
            )

        if self.bottlenecks:
            lines.extend(["", "Bottlenecks Detected:", "-" * 40])
            for bn in self.bottlenecks:
                lines.append(
                    f"  [{bn.severity.upper()}] {bn.state}: "
                    f"avg wait={_format_duration(bn.avg_wait_time)}"
                )
                lines.append(f"    → {bn.recommendation}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "avg_case_duration_seconds": self.avg_case_duration.total_seconds(),
            "throughput_per_day": self.throughput_per_day,
            "time_in_state": [s.to_dict() for s in self.time_in_state],
            "transitions": [t.to_dict() for t in self.transitions],
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
        }


class PerformanceAnalyzer:
    """
    Analyzes process performance from event logs.

    Computes time-in-state, identifies bottlenecks, and provides
    performance overlays for process visualization.
    """

    def __init__(
        self,
        state_extractor: Callable[[Event], str | None] | None = None,
        bottleneck_threshold: timedelta = timedelta(days=3),
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            state_extractor: Function to extract state from event.
                           Defaults to looking at STATE_CHANGED events.
            bottleneck_threshold: Time threshold to flag as bottleneck.
        """
        self.state_extractor = state_extractor or _default_state_extractor
        self.bottleneck_threshold = bottleneck_threshold

    def analyze(self, event_log: EventLog) -> PerformanceReport:
        """
        Perform complete performance analysis on an event log.

        Args:
            event_log: The event log to analyze

        Returns:
            PerformanceReport with all metrics
        """
        sorted_log = event_log.sorted_by_time()

        if not sorted_log.events:
            return self._empty_report()

        # Group events by case
        cases: dict[CaseId, list[Event]] = {}
        for event in sorted_log:
            if event.case_id not in cases:
                cases[event.case_id] = []
            cases[event.case_id].append(event)

        # Compute case durations
        case_durations = self._compute_case_durations(cases)

        # Compute time in state
        state_times = self._compute_time_in_state(cases)

        # Compute transitions
        transitions = self._compute_transitions(cases)

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(state_times)

        # Compute overall metrics
        total_cases = len(cases)
        durations = list(case_durations.values())

        avg_duration = timedelta(
            seconds=sum(d.total_seconds() for d in durations) / len(durations)
        ) if durations else timedelta(0)

        min_duration = min(durations) if durations else timedelta(0)
        max_duration = max(durations) if durations else timedelta(0)

        # Throughput
        time_range = sorted_log.time_range
        if time_range:
            total_days = (time_range[1] - time_range[0]).total_seconds() / 86400
            throughput = total_cases / total_days if total_days > 0 else 0.0
        else:
            throughput = 0.0

        return PerformanceReport(
            total_cases=total_cases,
            avg_case_duration=avg_duration,
            min_case_duration=min_duration,
            max_case_duration=max_duration,
            throughput_per_day=throughput,
            time_in_state=list(state_times.values()),
            transitions=list(transitions.values()),
            bottlenecks=bottlenecks,
            analysis_start=time_range[0] if time_range else datetime.now(),
            analysis_end=time_range[1] if time_range else datetime.now(),
        )

    def compute_time_in_state(
        self,
        event_log: EventLog,
    ) -> dict[str, TimeInStateStats]:
        """
        Compute time spent in each state.

        Args:
            event_log: The event log to analyze

        Returns:
            Dictionary mapping state names to TimeInStateStats
        """
        cases: dict[CaseId, list[Event]] = {}
        for event in event_log.sorted_by_time():
            if event.case_id not in cases:
                cases[event.case_id] = []
            cases[event.case_id].append(event)

        return self._compute_time_in_state(cases)

    def _compute_case_durations(
        self,
        cases: dict[CaseId, list[Event]],
    ) -> dict[CaseId, timedelta]:
        """Compute duration for each case."""
        durations = {}
        for case_id, events in cases.items():
            if len(events) >= 2:
                durations[case_id] = events[-1].timestamp - events[0].timestamp
            else:
                durations[case_id] = timedelta(0)
        return durations

    def _compute_time_in_state(
        self,
        cases: dict[CaseId, list[Event]],
    ) -> dict[str, TimeInStateStats]:
        """Compute time-in-state statistics."""
        # Collect all state durations
        state_durations: dict[str, list[tuple[CaseId, timedelta]]] = defaultdict(list)

        for case_id, events in cases.items():
            current_state: str | None = None
            state_entry_time: datetime | None = None

            for event in events:
                new_state = self.state_extractor(event)

                if new_state and new_state != current_state:
                    # State transition
                    if current_state and state_entry_time:
                        duration = event.timestamp - state_entry_time
                        state_durations[current_state].append((case_id, duration))

                    current_state = new_state
                    state_entry_time = event.timestamp

            # Handle final state (until last event)
            if current_state and state_entry_time and events:
                duration = events[-1].timestamp - state_entry_time
                state_durations[current_state].append((case_id, duration))

        # Build statistics
        stats: dict[str, TimeInStateStats] = {}

        for state, durations in state_durations.items():
            if not durations:
                continue

            times = [d for _, d in durations]
            times_sorted = sorted(times, key=lambda t: t.total_seconds())

            total_seconds = sum(t.total_seconds() for t in times)
            avg_seconds = total_seconds / len(times)

            # Percentiles
            median_idx = len(times_sorted) // 2
            p90_idx = int(len(times_sorted) * 0.9)

            # Longest cases
            longest = sorted(durations, key=lambda x: x[1].total_seconds(), reverse=True)[:5]

            stats[state] = TimeInStateStats(
                state=state,
                total_time=timedelta(seconds=total_seconds),
                case_count=len(set(cid for cid, _ in durations)),
                entry_count=len(durations),
                min_time=times_sorted[0],
                max_time=times_sorted[-1],
                avg_time=timedelta(seconds=avg_seconds),
                median_time=times_sorted[median_idx] if times_sorted else None,
                p90_time=times_sorted[p90_idx] if len(times_sorted) > p90_idx else None,
                longest_cases=longest,
            )

        return stats

    def _compute_transitions(
        self,
        cases: dict[CaseId, list[Event]],
    ) -> dict[tuple[str, str], TransitionStats]:
        """Compute transition statistics."""
        transition_times: dict[tuple[str, str], list[timedelta]] = defaultdict(list)

        for case_id, events in cases.items():
            current_state: str | None = None
            state_entry_time: datetime | None = None

            for event in events:
                new_state = self.state_extractor(event)

                if new_state and new_state != current_state:
                    if current_state and state_entry_time:
                        key = (current_state, new_state)
                        duration = event.timestamp - state_entry_time
                        transition_times[key].append(duration)

                    current_state = new_state
                    state_entry_time = event.timestamp

        # Build statistics
        stats: dict[tuple[str, str], TransitionStats] = {}

        for (from_state, to_state), times in transition_times.items():
            if not times:
                continue

            total_seconds = sum(t.total_seconds() for t in times)
            avg_seconds = total_seconds / len(times)

            stats[(from_state, to_state)] = TransitionStats(
                from_state=from_state,
                to_state=to_state,
                count=len(times),
                avg_time=timedelta(seconds=avg_seconds),
                min_time=min(times),
                max_time=max(times),
            )

        return stats

    def _detect_bottlenecks(
        self,
        state_stats: dict[str, TimeInStateStats],
    ) -> list[BottleneckInfo]:
        """Detect process bottlenecks."""
        bottlenecks = []
        threshold_seconds = self.bottleneck_threshold.total_seconds()

        # Sort states by avg time
        sorted_states = sorted(
            state_stats.values(),
            key=lambda s: s.avg_time.total_seconds(),
            reverse=True,
        )

        total_time = sum(s.avg_time.total_seconds() for s in sorted_states)

        for stats in sorted_states:
            avg_seconds = stats.avg_time.total_seconds()

            if avg_seconds >= threshold_seconds:
                # Calculate impact score (proportion of total time)
                impact = avg_seconds / total_time if total_time > 0 else 0

                # Determine severity
                if avg_seconds >= threshold_seconds * 2:
                    severity = "high"
                elif avg_seconds >= threshold_seconds * 1.5:
                    severity = "medium"
                else:
                    severity = "low"

                # Generate recommendation
                recommendation = self._generate_recommendation(stats, severity)

                bottlenecks.append(
                    BottleneckInfo(
                        state=stats.state,
                        severity=severity,
                        avg_wait_time=stats.avg_time,
                        impact_score=impact,
                        affected_cases=stats.case_count,
                        recommendation=recommendation,
                    )
                )

        return bottlenecks

    def _generate_recommendation(
        self,
        stats: TimeInStateStats,
        severity: str,
    ) -> str:
        """Generate a recommendation for addressing a bottleneck."""
        state = stats.state.lower()

        if "wait" in state or "pending" in state:
            return "Consider automating follow-up or reducing wait time thresholds"
        elif "review" in state or "approval" in state:
            return "Consider parallel review paths or delegation rules"
        elif "contact" in state:
            return "Consider multi-channel outreach or escalation triggers"
        elif severity == "high":
            return "Investigate root cause; consider process redesign"
        else:
            return "Monitor and set SLA alerts"

    def _empty_report(self) -> PerformanceReport:
        """Return an empty report for empty event logs."""
        now = datetime.now()
        return PerformanceReport(
            total_cases=0,
            avg_case_duration=timedelta(0),
            min_case_duration=timedelta(0),
            max_case_duration=timedelta(0),
            throughput_per_day=0.0,
            time_in_state=[],
            transitions=[],
            bottlenecks=[],
            analysis_start=now,
            analysis_end=now,
        )


def _default_state_extractor(event: Event) -> str | None:
    """Default function to extract state from event."""
    if event.event_type == EventType.STATE_CHANGED:
        return event.payload.get("new_state") or event.payload.get("status")
    elif event.event_type == EventType.ENTITY_CREATED:
        return "created"
    elif event.event_type == EventType.ENTITY_CLOSED:
        return "closed"
    return None


def _format_duration(td: timedelta) -> str:
    """Format a timedelta for display."""
    total_seconds = int(td.total_seconds())

    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes}m"
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        return f"{days}d {hours}h"
