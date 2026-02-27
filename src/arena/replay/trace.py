"""Execution trace capture for replay and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json

from arena.decision.extractor import DecisionPoint
from arena.policy.result import ConstraintViolation, PolicyResult


@dataclass
class ExecutionTrace:
    """
    Record of a single policy decision execution.

    Captures everything needed to understand and evaluate the decision.
    """

    # Decision point info
    decision_point: DecisionPoint

    # Policy result
    result: PolicyResult

    # Timing
    executed_at: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0

    # Policy metadata
    policy_name: str = ""
    policy_version: str = ""

    # Validation
    violations: list[ConstraintViolation] = field(default_factory=list)

    # Comparison to historical (if available)
    matched_historical: bool = False
    historical_action_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "decision_point_id": self.decision_point.decision_id,
            "case_id": str(self.decision_point.case_id),
            "timestamp": self.decision_point.timestamp.isoformat(),
            "rule_name": self.decision_point.rule_name,
            "snapshot_summary": self.decision_point.snapshot.to_dict(),
            "allowed_actions": [str(a) for a in self.decision_point.allowed_actions],
            "result": self.result.to_dict(),
            "executed_at": self.executed_at.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "violations": [
                {
                    "constraint_name": v.constraint_name,
                    "severity": v.severity,
                    "message": v.message,
                }
                for v in self.violations
            ],
            "matched_historical": self.matched_historical,
            "historical_action_type": self.historical_action_type,
            "outcome_label": self.decision_point.outcome_label,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class TraceCollection:
    """Collection of execution traces with query methods."""

    traces: list[ExecutionTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self):
        return iter(self.traces)

    def append(self, trace: ExecutionTrace) -> None:
        """Add a trace to the collection."""
        self.traces.append(trace)

    def filter_by_case(self, case_id: str) -> TraceCollection:
        """Get traces for a specific case."""
        filtered = [
            t for t in self.traces if str(t.decision_point.case_id) == case_id
        ]
        return TraceCollection(traces=filtered, metadata=self.metadata)

    def filter_by_rule(self, rule_name: str) -> TraceCollection:
        """Get traces for a specific rule."""
        filtered = [
            t for t in self.traces if t.decision_point.rule_name == rule_name
        ]
        return TraceCollection(traces=filtered, metadata=self.metadata)

    def filter_by_action(self, action_type: str) -> TraceCollection:
        """Get traces where a specific action was chosen."""
        filtered = [
            t for t in self.traces if t.result.action.action_type == action_type
        ]
        return TraceCollection(traces=filtered, metadata=self.metadata)

    def filter_by_outcome(self, outcome_label: str) -> TraceCollection:
        """Get traces with a specific outcome."""
        filtered = [
            t
            for t in self.traces
            if t.decision_point.outcome_label == outcome_label
        ]
        return TraceCollection(traces=filtered, metadata=self.metadata)

    def with_violations(self) -> TraceCollection:
        """Get traces that have constraint violations."""
        filtered = [t for t in self.traces if t.violations]
        return TraceCollection(traces=filtered, metadata=self.metadata)

    def matched_historical(self) -> TraceCollection:
        """Get traces where the policy matched historical action."""
        filtered = [t for t in self.traces if t.matched_historical]
        return TraceCollection(traces=filtered, metadata=self.metadata)

    @property
    def action_distribution(self) -> dict[str, int]:
        """Get distribution of chosen actions."""
        dist: dict[str, int] = {}
        for trace in self.traces:
            action_type = trace.result.action.action_type
            dist[action_type] = dist.get(action_type, 0) + 1
        return dist

    @property
    def outcome_distribution(self) -> dict[str, int]:
        """Get distribution of outcomes."""
        dist: dict[str, int] = {}
        for trace in self.traces:
            outcome = trace.decision_point.outcome_label or "unknown"
            dist[outcome] = dist.get(outcome, 0) + 1
        return dist

    @property
    def violation_count(self) -> int:
        """Total number of violations across all traces."""
        return sum(len(t.violations) for t in self.traces)

    @property
    def historical_match_rate(self) -> float:
        """Rate of matching historical actions."""
        if not self.traces:
            return 0.0
        matched = sum(1 for t in self.traces if t.matched_historical)
        return matched / len(self.traces)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all traces to dictionaries."""
        return [t.to_dict() for t in self.traces]

    def to_jsonl(self, path: str) -> None:
        """Write traces to JSONL file."""
        with open(path, "w") as f:
            for trace in self.traces:
                f.write(trace.to_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> TraceCollection:
        """Load traces from JSONL file."""
        # Note: This is a simplified loader; full deserialization
        # would need to reconstruct DecisionPoint and PolicyResult
        traces = []
        metadata = {"source_file": path}
        # For now, just return empty - full implementation would parse
        return cls(traces=traces, metadata=metadata)

    def summary(self) -> dict[str, Any]:
        """Get a summary of the trace collection."""
        return {
            "total_traces": len(self.traces),
            "unique_cases": len(set(str(t.decision_point.case_id) for t in self.traces)),
            "action_distribution": self.action_distribution,
            "outcome_distribution": self.outcome_distribution,
            "violation_count": self.violation_count,
            "historical_match_rate": self.historical_match_rate,
        }
