"""Variant analysis for comparing process trajectories.

This module provides tools for analyzing and comparing variants
(different paths through a process) to understand what distinguishes
successful vs unsuccessful cases.

Based on Apromore's variant analysis capabilities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Sequence

from arena.core.event import Event, EventLog, EventType
from arena.core.types import CaseId


@dataclass
class Variant:
    """
    A process variant - a unique sequence of activities.

    Variants are the distinct paths that cases take through a process.
    Comparing variants helps identify what distinguishes successful
    from unsuccessful outcomes.
    """

    # The activity sequence that defines this variant
    activity_sequence: tuple[str, ...]

    # Cases that follow this variant
    case_ids: list[CaseId] = field(default_factory=list)

    # Statistics
    frequency: int = 0
    avg_duration: timedelta | None = None
    min_duration: timedelta | None = None
    max_duration: timedelta | None = None

    # Outcome distribution for cases in this variant
    outcome_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        """Get a string signature for this variant."""
        return " -> ".join(self.activity_sequence)

    @property
    def positive_rate(self) -> float:
        """Rate of positive outcomes in this variant."""
        total = sum(self.outcome_distribution.values())
        if total == 0:
            return 0.0
        positive = self.outcome_distribution.get("positive", 0)
        return positive / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "activity_sequence": list(self.activity_sequence),
            "frequency": self.frequency,
            "case_count": len(self.case_ids),
            "avg_duration_seconds": (
                self.avg_duration.total_seconds() if self.avg_duration else None
            ),
            "outcome_distribution": self.outcome_distribution,
            "positive_rate": self.positive_rate,
        }


@dataclass
class VariantComparison:
    """Comparison between two sets of variants (e.g., successful vs unsuccessful)."""

    group_a_name: str
    group_b_name: str

    # Variants unique to each group
    unique_to_a: list[Variant] = field(default_factory=list)
    unique_to_b: list[Variant] = field(default_factory=list)

    # Variants shared by both groups
    shared: list[tuple[Variant, Variant]] = field(default_factory=list)

    # Statistical summary
    avg_length_a: float = 0.0
    avg_length_b: float = 0.0
    avg_duration_a: timedelta | None = None
    avg_duration_b: timedelta | None = None

    def summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = [
            f"Variant Comparison: {self.group_a_name} vs {self.group_b_name}",
            "=" * 60,
            "",
            f"Variants unique to {self.group_a_name}: {len(self.unique_to_a)}",
            f"Variants unique to {self.group_b_name}: {len(self.unique_to_b)}",
            f"Shared variants: {len(self.shared)}",
            "",
            f"Avg sequence length ({self.group_a_name}): {self.avg_length_a:.1f}",
            f"Avg sequence length ({self.group_b_name}): {self.avg_length_b:.1f}",
        ]

        if self.unique_to_a:
            lines.append(f"\nTop variants unique to {self.group_a_name}:")
            for v in self.unique_to_a[:5]:
                lines.append(f"  {v.frequency}x: {v.signature}")

        if self.unique_to_b:
            lines.append(f"\nTop variants unique to {self.group_b_name}:")
            for v in self.unique_to_b[:5]:
                lines.append(f"  {v.frequency}x: {v.signature}")

        return "\n".join(lines)


class VariantAnalyzer:
    """
    Analyzes process variants in event logs.

    Variants are distinct sequences of activities that cases follow.
    This analyzer extracts variants, computes statistics, and enables
    comparison between different case populations.
    """

    def __init__(
        self,
        activity_extractor: Callable[[Event], str] | None = None,
        outcome_extractor: Callable[[list[Event]], str | None] | None = None,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            activity_extractor: Function to extract activity name from event.
                               Defaults to using event_type + payload hints.
            outcome_extractor: Function to extract outcome from case events.
                              Defaults to looking for OUTCOME_RECORDED events.
        """
        self.activity_extractor = activity_extractor or _default_activity_extractor
        self.outcome_extractor = outcome_extractor or _default_outcome_extractor

    def extract_variants(
        self,
        event_log: EventLog,
        min_frequency: int = 1,
    ) -> list[Variant]:
        """
        Extract all variants from an event log.

        Args:
            event_log: The event log to analyze
            min_frequency: Minimum frequency to include a variant

        Returns:
            List of Variant objects, sorted by frequency (descending)
        """
        # Group events by case
        cases: dict[CaseId, list[Event]] = {}
        for event in event_log.sorted_by_time():
            if event.case_id not in cases:
                cases[event.case_id] = []
            cases[event.case_id].append(event)

        # Extract activity sequences per case
        case_sequences: dict[CaseId, tuple[str, ...]] = {}
        case_outcomes: dict[CaseId, str | None] = {}
        case_durations: dict[CaseId, timedelta] = {}

        for case_id, events in cases.items():
            # Extract activity sequence
            activities = tuple(self.activity_extractor(e) for e in events)
            case_sequences[case_id] = activities

            # Extract outcome
            case_outcomes[case_id] = self.outcome_extractor(events)

            # Compute duration
            if len(events) >= 2:
                case_durations[case_id] = events[-1].timestamp - events[0].timestamp

        # Group cases by sequence (variant)
        variant_cases: dict[tuple[str, ...], list[CaseId]] = {}
        for case_id, sequence in case_sequences.items():
            if sequence not in variant_cases:
                variant_cases[sequence] = []
            variant_cases[sequence].append(case_id)

        # Build Variant objects
        variants = []
        for sequence, case_ids in variant_cases.items():
            if len(case_ids) < min_frequency:
                continue

            # Compute outcome distribution
            outcome_dist: dict[str, int] = {}
            for cid in case_ids:
                outcome = case_outcomes.get(cid) or "unknown"
                outcome_dist[outcome] = outcome_dist.get(outcome, 0) + 1

            # Compute duration stats
            durations = [case_durations[cid] for cid in case_ids if cid in case_durations]
            avg_duration = None
            min_duration = None
            max_duration = None
            if durations:
                total_seconds = sum(d.total_seconds() for d in durations)
                avg_duration = timedelta(seconds=total_seconds / len(durations))
                min_duration = min(durations)
                max_duration = max(durations)

            variants.append(
                Variant(
                    activity_sequence=sequence,
                    case_ids=case_ids,
                    frequency=len(case_ids),
                    avg_duration=avg_duration,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    outcome_distribution=outcome_dist,
                )
            )

        # Sort by frequency
        variants.sort(key=lambda v: -v.frequency)

        return variants

    def compare_groups(
        self,
        event_log: EventLog,
        group_a_filter: Callable[[CaseId, list[Event]], bool],
        group_b_filter: Callable[[CaseId, list[Event]], bool],
        group_a_name: str = "Group A",
        group_b_name: str = "Group B",
    ) -> VariantComparison:
        """
        Compare variants between two groups of cases.

        Useful for comparing:
        - Successful vs unsuccessful outcomes
        - Different time periods
        - Different customer segments
        - Different agent behaviors

        Args:
            event_log: The event log to analyze
            group_a_filter: Function to determine if case belongs to group A
            group_b_filter: Function to determine if case belongs to group B
            group_a_name: Name for group A (for reporting)
            group_b_name: Name for group B (for reporting)

        Returns:
            VariantComparison with analysis results
        """
        # Split event log by group
        cases: dict[CaseId, list[Event]] = {}
        for event in event_log.sorted_by_time():
            if event.case_id not in cases:
                cases[event.case_id] = []
            cases[event.case_id].append(event)

        group_a_events = []
        group_b_events = []

        for case_id, events in cases.items():
            if group_a_filter(case_id, events):
                group_a_events.extend(events)
            elif group_b_filter(case_id, events):
                group_b_events.extend(events)

        # Extract variants for each group
        variants_a = self.extract_variants(EventLog(group_a_events))
        variants_b = self.extract_variants(EventLog(group_b_events))

        # Build signature -> variant maps
        map_a = {v.signature: v for v in variants_a}
        map_b = {v.signature: v for v in variants_b}

        # Find unique and shared variants
        unique_to_a = [v for v in variants_a if v.signature not in map_b]
        unique_to_b = [v for v in variants_b if v.signature not in map_a]
        shared = [
            (v, map_b[v.signature])
            for v in variants_a
            if v.signature in map_b
        ]

        # Compute average lengths
        avg_length_a = (
            sum(len(v.activity_sequence) * v.frequency for v in variants_a)
            / sum(v.frequency for v in variants_a)
            if variants_a else 0.0
        )
        avg_length_b = (
            sum(len(v.activity_sequence) * v.frequency for v in variants_b)
            / sum(v.frequency for v in variants_b)
            if variants_b else 0.0
        )

        # Compute average durations
        def weighted_avg_duration(variants: list[Variant]) -> timedelta | None:
            total_seconds = 0.0
            total_count = 0
            for v in variants:
                if v.avg_duration:
                    total_seconds += v.avg_duration.total_seconds() * v.frequency
                    total_count += v.frequency
            if total_count == 0:
                return None
            return timedelta(seconds=total_seconds / total_count)

        return VariantComparison(
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            unique_to_a=unique_to_a,
            unique_to_b=unique_to_b,
            shared=shared,
            avg_length_a=avg_length_a,
            avg_length_b=avg_length_b,
            avg_duration_a=weighted_avg_duration(variants_a),
            avg_duration_b=weighted_avg_duration(variants_b),
        )


def compare_variants(
    event_log: EventLog,
    positive_outcomes: Sequence[str] | None = None,
    negative_outcomes: Sequence[str] | None = None,
) -> VariantComparison:
    """
    Compare variants between successful and unsuccessful cases.

    Convenience function for the most common comparison: positive vs negative outcomes.

    Args:
        event_log: The event log to analyze
        positive_outcomes: List of outcome labels considered successful.
                          Defaults to common positive outcomes.
        negative_outcomes: List of outcome labels considered unsuccessful.
                          Defaults to common negative outcomes.

    Returns:
        VariantComparison between successful and unsuccessful cases
    """
    if positive_outcomes is None:
        positive_outcomes = [
            "meeting_booked",
            "deal_closed_won",
            "reply_received",
            "visit_completed",
            "appointment_booked",
        ]

    if negative_outcomes is None:
        negative_outcomes = [
            "lead_went_cold",
            "deal_closed_lost",
            "no_response",
            "lost_to_followup",
            "unsubscribed",
        ]

    positive_set = set(positive_outcomes)
    negative_set = set(negative_outcomes)

    def is_positive(case_id: CaseId, events: list[Event]) -> bool:
        for event in events:
            if event.event_type == EventType.OUTCOME_RECORDED:
                outcome = event.payload.get("outcome")
                if outcome in positive_set:
                    return True
        return False

    def is_negative(case_id: CaseId, events: list[Event]) -> bool:
        for event in events:
            if event.event_type == EventType.OUTCOME_RECORDED:
                outcome = event.payload.get("outcome")
                if outcome in negative_set:
                    return True
        return False

    analyzer = VariantAnalyzer()
    return analyzer.compare_groups(
        event_log,
        group_a_filter=is_positive,
        group_b_filter=is_negative,
        group_a_name="Successful",
        group_b_name="Unsuccessful",
    )


def _default_activity_extractor(event: Event) -> str:
    """Default function to extract activity name from event."""
    base = event.event_type.value

    # Add context from payload
    if event.event_type == EventType.INTERACTION_OCCURRED:
        direction = event.payload.get("direction", "")
        if direction:
            return f"{base}:{direction}"

    elif event.event_type == EventType.STATE_CHANGED:
        new_state = event.payload.get("new_state", "")
        if new_state:
            return f"{base}:{new_state}"

    elif event.event_type == EventType.OUTCOME_RECORDED:
        outcome = event.payload.get("outcome", "")
        if outcome:
            return f"{base}:{outcome}"

    return base


def _default_outcome_extractor(events: list[Event]) -> str | None:
    """Default function to extract outcome from case events."""
    positive_outcomes = {
        "meeting_booked", "deal_closed_won", "reply_received",
        "visit_completed", "appointment_booked",
    }
    negative_outcomes = {
        "lead_went_cold", "deal_closed_lost", "no_response",
        "lost_to_followup", "unsubscribed",
    }

    for event in events:
        if event.event_type == EventType.OUTCOME_RECORDED:
            outcome = event.payload.get("outcome")
            if outcome:
                if outcome in positive_outcomes:
                    return "positive"
                elif outcome in negative_outcomes:
                    return "negative"
                else:
                    return "neutral"
    return None
