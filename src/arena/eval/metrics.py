"""Evaluation metrics for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from arena.decision.extractor import DecisionPoint
from arena.decision.labeler import OutcomeCategory
from arena.replay.trace import ExecutionTrace, TraceCollection


@dataclass
class MetricResult:
    """Result of computing a metric."""

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


@dataclass
class MetricsSummary:
    """Summary of all computed metrics."""

    metrics: dict[str, MetricResult] = field(default_factory=dict)

    def __getitem__(self, key: str) -> MetricResult:
        return self.metrics[key]

    def get(self, key: str, default: float = 0.0) -> float:
        """Get a metric value by name."""
        if key in self.metrics:
            return self.metrics[key].value
        return default

    def add(self, metric: MetricResult) -> None:
        """Add a metric to the summary."""
        self.metrics[metric.name] = metric

    def to_dict(self) -> dict[str, float]:
        """Get metrics as a simple dictionary."""
        return {name: result.value for name, result in self.metrics.items()}


def compute_metrics(
    traces: TraceCollection,
    decision_points: Sequence[DecisionPoint] | None = None,
) -> MetricsSummary:
    """
    Compute all standard metrics from execution traces.

    Args:
        traces: Collection of execution traces
        decision_points: Original decision points (for outcome labels)

    Returns:
        MetricsSummary with all computed metrics
    """
    summary = MetricsSummary()

    # Basic counts
    summary.add(MetricResult(
        name="total_decisions",
        value=float(len(traces)),
    ))

    # Violation metrics
    summary.add(compute_violation_rate(traces))

    # Historical match rate
    summary.add(compute_historical_match_rate(traces))

    # Outcome-based metrics (if outcomes are labeled)
    if _has_outcomes(traces):
        summary.add(compute_positive_outcome_rate(traces))
        summary.add(compute_conversion_rate(traces))

    # Action diversity
    summary.add(compute_action_entropy(traces))

    # Confidence metrics
    summary.add(compute_average_confidence(traces))

    return summary


def compute_precision_at_k(
    traces: TraceCollection,
    k: int,
    relevance_fn: Callable[[ExecutionTrace], bool] | None = None,
) -> MetricResult:
    """
    Compute Precision@K for ranked decision recommendations.

    Args:
        traces: Traces sorted by relevance/priority
        k: Number of top items to consider
        relevance_fn: Function to determine if a trace is "relevant"
                     Defaults to checking for positive outcomes

    Returns:
        MetricResult with precision value
    """
    if not traces.traces:
        return MetricResult(name=f"precision@{k}", value=0.0)

    if relevance_fn is None:
        relevance_fn = lambda t: t.decision_point.outcome_label in [
            "meeting_booked",
            "deal_closed_won",
            "reply_received",
        ]

    top_k = traces.traces[:k]
    relevant = sum(1 for t in top_k if relevance_fn(t))

    return MetricResult(
        name=f"precision@{k}",
        value=relevant / k if k > 0 else 0.0,
        metadata={"k": k, "relevant_count": relevant},
    )


def compute_ndcg_at_k(
    traces: TraceCollection,
    k: int,
    gain_fn: Callable[[ExecutionTrace], float] | None = None,
) -> MetricResult:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        traces: Traces sorted by predicted relevance
        k: Number of top items to consider
        gain_fn: Function to compute the gain (relevance) of each trace
                 Defaults to binary gain based on positive outcomes

    Returns:
        MetricResult with NDCG value
    """
    import math

    if not traces.traces:
        return MetricResult(name=f"ndcg@{k}", value=0.0)

    if gain_fn is None:
        def gain_fn(t: ExecutionTrace) -> float:
            label = t.decision_point.outcome_label
            if label in ["deal_closed_won"]:
                return 3.0
            elif label in ["meeting_booked"]:
                return 2.0
            elif label in ["reply_received"]:
                return 1.0
            return 0.0

    # Compute DCG
    dcg = 0.0
    for i, trace in enumerate(traces.traces[:k]):
        gain = gain_fn(trace)
        dcg += gain / math.log2(i + 2)  # log2(position + 1), 1-indexed

    # Compute ideal DCG (sort by gain)
    gains = [gain_fn(t) for t in traces.traces[:k]]
    gains.sort(reverse=True)

    idcg = 0.0
    for i, gain in enumerate(gains):
        idcg += gain / math.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return MetricResult(
        name=f"ndcg@{k}",
        value=ndcg,
        metadata={"k": k, "dcg": dcg, "idcg": idcg},
    )


def compute_violation_rate(traces: TraceCollection) -> MetricResult:
    """Compute the rate of constraint violations."""
    if not traces.traces:
        return MetricResult(name="violation_rate", value=0.0)

    violated = sum(1 for t in traces if t.violations)
    return MetricResult(
        name="violation_rate",
        value=violated / len(traces),
        metadata={"violated_count": violated, "total": len(traces)},
    )


def compute_historical_match_rate(traces: TraceCollection) -> MetricResult:
    """Compute how often the policy matched historical actions."""
    if not traces.traces:
        return MetricResult(name="historical_match_rate", value=0.0)

    # Only count traces where historical action is known
    with_historical = [t for t in traces if t.historical_action_type is not None]

    if not with_historical:
        return MetricResult(
            name="historical_match_rate",
            value=0.0,
            metadata={"no_historical_data": True},
        )

    matched = sum(1 for t in with_historical if t.matched_historical)

    return MetricResult(
        name="historical_match_rate",
        value=matched / len(with_historical),
        metadata={"matched": matched, "total": len(with_historical)},
    )


def compute_positive_outcome_rate(traces: TraceCollection) -> MetricResult:
    """Compute the rate of positive outcomes."""
    if not traces.traces:
        return MetricResult(name="positive_outcome_rate", value=0.0)

    labeled = [t for t in traces if t.decision_point.outcome_label]

    if not labeled:
        return MetricResult(
            name="positive_outcome_rate",
            value=0.0,
            metadata={"no_labeled_outcomes": True},
        )

    positive_labels = {
        "meeting_booked",
        "deal_closed_won",
        "reply_received",
        "visit_completed",
        "appointment_booked",
    }

    positive = sum(
        1 for t in labeled if t.decision_point.outcome_label in positive_labels
    )

    return MetricResult(
        name="positive_outcome_rate",
        value=positive / len(labeled),
        metadata={"positive": positive, "labeled": len(labeled)},
    )


def compute_conversion_rate(
    traces: TraceCollection,
    target_outcome: str = "deal_closed_won",
) -> MetricResult:
    """Compute conversion rate to a specific outcome."""
    if not traces.traces:
        return MetricResult(name="conversion_rate", value=0.0)

    labeled = [t for t in traces if t.decision_point.outcome_label]

    if not labeled:
        return MetricResult(
            name="conversion_rate",
            value=0.0,
            metadata={"no_labeled_outcomes": True},
        )

    converted = sum(
        1 for t in labeled if t.decision_point.outcome_label == target_outcome
    )

    return MetricResult(
        name="conversion_rate",
        value=converted / len(labeled),
        metadata={
            "converted": converted,
            "total": len(labeled),
            "target_outcome": target_outcome,
        },
    )


def compute_action_entropy(traces: TraceCollection) -> MetricResult:
    """
    Compute entropy of action distribution.

    Higher entropy means more diverse action selection.
    """
    import math

    if not traces.traces:
        return MetricResult(name="action_entropy", value=0.0)

    # Count actions
    action_counts: dict[str, int] = {}
    for trace in traces:
        action_type = trace.result.action.action_type
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    # Compute entropy
    total = len(traces)
    entropy = 0.0

    for count in action_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize by max possible entropy
    max_entropy = math.log2(len(action_counts)) if len(action_counts) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return MetricResult(
        name="action_entropy",
        value=normalized_entropy,
        metadata={
            "raw_entropy": entropy,
            "action_distribution": action_counts,
            "unique_actions": len(action_counts),
        },
    )


def compute_average_confidence(traces: TraceCollection) -> MetricResult:
    """Compute average confidence of policy decisions."""
    if not traces.traces:
        return MetricResult(name="average_confidence", value=0.0)

    confidences = [
        t.result.confidence
        for t in traces
        if t.result.confidence is not None
    ]

    if not confidences:
        return MetricResult(
            name="average_confidence",
            value=0.0,
            metadata={"no_confidence_values": True},
        )

    return MetricResult(
        name="average_confidence",
        value=sum(confidences) / len(confidences),
        metadata={
            "min": min(confidences),
            "max": max(confidences),
            "count": len(confidences),
        },
    )


def _has_outcomes(traces: TraceCollection) -> bool:
    """Check if traces have outcome labels."""
    return any(t.decision_point.outcome_label for t in traces)
