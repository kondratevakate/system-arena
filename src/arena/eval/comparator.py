"""Policy comparison and uplift analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from arena.eval.metrics import MetricsSummary, compute_metrics
from arena.replay.trace import ExecutionTrace, TraceCollection


@dataclass
class ComparisonResult:
    """Result of comparing two policies."""

    policy_name: str
    baseline_name: str

    # Metrics for each
    policy_metrics: MetricsSummary
    baseline_metrics: MetricsSummary

    # Differences (policy - baseline)
    metric_diffs: dict[str, float] = field(default_factory=dict)

    # Uplift percentages
    uplift: dict[str, float] = field(default_factory=dict)

    # Statistical significance (if computed)
    p_values: dict[str, float] = field(default_factory=dict)

    # Per-decision agreement
    agreement_rate: float = 0.0
    disagreements: list[DisagreementRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "policy_name": self.policy_name,
            "baseline_name": self.baseline_name,
            "policy_metrics": self.policy_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "metric_diffs": self.metric_diffs,
            "uplift": self.uplift,
            "agreement_rate": self.agreement_rate,
            "total_disagreements": len(self.disagreements),
        }

    def summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = [
            f"Policy Comparison: {self.policy_name} vs {self.baseline_name}",
            "=" * 60,
            "",
            "Metric Differences (policy - baseline):",
        ]

        for metric, diff in self.metric_diffs.items():
            uplift_pct = self.uplift.get(metric, 0.0) * 100
            sign = "+" if diff >= 0 else ""
            lines.append(f"  {metric}: {sign}{diff:.4f} ({sign}{uplift_pct:.1f}%)")

        lines.extend([
            "",
            f"Agreement Rate: {self.agreement_rate:.1%}",
            f"Total Disagreements: {len(self.disagreements)}",
        ])

        return "\n".join(lines)


@dataclass
class DisagreementRecord:
    """Record of a decision where policy and baseline disagreed."""

    decision_point_id: str
    case_id: str
    policy_action: str
    baseline_action: str
    policy_confidence: float | None
    baseline_confidence: float | None
    outcome_label: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyComparator:
    """
    Compares policy performance against a baseline.

    Computes uplift metrics and identifies where policies differ.
    """

    def __init__(
        self,
        significance_test: Callable[[list[float], list[float]], float] | None = None,
    ) -> None:
        """
        Initialize the comparator.

        Args:
            significance_test: Optional function for computing p-values.
                              Takes two lists of values, returns p-value.
        """
        self.significance_test = significance_test

    def compare(
        self,
        policy_traces: TraceCollection,
        baseline_traces: TraceCollection,
        policy_name: str = "policy",
        baseline_name: str = "baseline",
    ) -> ComparisonResult:
        """
        Compare a policy against a baseline.

        Traces must be aligned (same decision points in same order).
        """
        # Compute metrics for each
        policy_metrics = compute_metrics(policy_traces)
        baseline_metrics = compute_metrics(baseline_traces)

        # Compute differences and uplift
        metric_diffs = {}
        uplift = {}

        for metric_name in policy_metrics.metrics:
            policy_val = policy_metrics.get(metric_name)
            baseline_val = baseline_metrics.get(metric_name)

            diff = policy_val - baseline_val
            metric_diffs[metric_name] = diff

            # Compute relative uplift (avoid division by zero)
            if baseline_val != 0:
                uplift[metric_name] = diff / abs(baseline_val)
            elif policy_val != 0:
                uplift[metric_name] = 1.0  # Infinite improvement
            else:
                uplift[metric_name] = 0.0

        # Compute agreement rate and find disagreements
        agreement_rate, disagreements = self._compute_agreement(
            policy_traces, baseline_traces
        )

        # Compute statistical significance if test function provided
        p_values = {}
        # (Would require per-decision values, omitted for simplicity)

        return ComparisonResult(
            policy_name=policy_name,
            baseline_name=baseline_name,
            policy_metrics=policy_metrics,
            baseline_metrics=baseline_metrics,
            metric_diffs=metric_diffs,
            uplift=uplift,
            p_values=p_values,
            agreement_rate=agreement_rate,
            disagreements=disagreements,
        )

    def _compute_agreement(
        self,
        policy_traces: TraceCollection,
        baseline_traces: TraceCollection,
    ) -> tuple[float, list[DisagreementRecord]]:
        """Compute agreement rate and identify disagreements."""
        if len(policy_traces) != len(baseline_traces):
            # Can't compare if lengths differ
            return 0.0, []

        agreements = 0
        disagreements = []

        for p_trace, b_trace in zip(policy_traces.traces, baseline_traces.traces):
            p_action = p_trace.result.action.action_type
            b_action = b_trace.result.action.action_type

            if p_action == b_action:
                agreements += 1
            else:
                disagreements.append(
                    DisagreementRecord(
                        decision_point_id=p_trace.decision_point.decision_id,
                        case_id=str(p_trace.decision_point.case_id),
                        policy_action=p_action,
                        baseline_action=b_action,
                        policy_confidence=p_trace.result.confidence,
                        baseline_confidence=b_trace.result.confidence,
                        outcome_label=p_trace.decision_point.outcome_label,
                    )
                )

        agreement_rate = agreements / len(policy_traces) if policy_traces.traces else 0.0
        return agreement_rate, disagreements


def compare_policies(
    policy_traces: TraceCollection,
    baseline_traces: TraceCollection,
    policy_name: str = "policy",
    baseline_name: str = "baseline",
) -> ComparisonResult:
    """
    Convenience function to compare two policies.

    Args:
        policy_traces: Traces from the policy being evaluated
        baseline_traces: Traces from the baseline policy
        policy_name: Name for the evaluated policy
        baseline_name: Name for the baseline

    Returns:
        ComparisonResult with metrics and analysis
    """
    comparator = PolicyComparator()
    return comparator.compare(
        policy_traces, baseline_traces, policy_name, baseline_name
    )


def compute_uplift_by_segment(
    policy_traces: TraceCollection,
    baseline_traces: TraceCollection,
    segment_fn: Callable[[ExecutionTrace], str],
) -> dict[str, ComparisonResult]:
    """
    Compute uplift broken down by segments.

    Useful for understanding where a policy performs better/worse.

    Args:
        policy_traces: Traces from the policy
        baseline_traces: Traces from the baseline
        segment_fn: Function to determine segment for each trace

    Returns:
        Dictionary mapping segment names to comparison results
    """
    # Group traces by segment
    policy_segments: dict[str, list[ExecutionTrace]] = {}
    baseline_segments: dict[str, list[ExecutionTrace]] = {}

    for trace in policy_traces:
        segment = segment_fn(trace)
        if segment not in policy_segments:
            policy_segments[segment] = []
        policy_segments[segment].append(trace)

    for trace in baseline_traces:
        segment = segment_fn(trace)
        if segment not in baseline_segments:
            baseline_segments[segment] = []
        baseline_segments[segment].append(trace)

    # Compare each segment
    results = {}
    all_segments = set(policy_segments.keys()) | set(baseline_segments.keys())

    comparator = PolicyComparator()

    for segment in all_segments:
        p_traces = TraceCollection(traces=policy_segments.get(segment, []))
        b_traces = TraceCollection(traces=baseline_segments.get(segment, []))

        if p_traces.traces and b_traces.traces:
            results[segment] = comparator.compare(
                p_traces, b_traces, "policy", "baseline"
            )

    return results
