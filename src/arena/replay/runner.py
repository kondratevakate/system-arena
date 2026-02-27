"""Main replay runner for benchmark execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from arena.core.event import EventLog
from arena.decision.extractor import DecisionPoint, DecisionPointRule, ExtractionResult, extract_decision_points
from arena.policy.base import Policy
from arena.policy.result import ConstraintViolation, PolicyResult
from arena.replay.timeline import ReplayContext, Timeline
from arena.replay.trace import ExecutionTrace, TraceCollection


@dataclass
class Constraint:
    """A constraint that must not be violated during replay."""

    name: str
    rule: str  # Python expression
    severity: str = "violation"  # "warning", "violation", "critical"
    message: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Decision point extraction
    decision_rules: list[DecisionPointRule] = field(default_factory=list)

    # Constraints
    constraints: list[Constraint] = field(default_factory=list)

    # Options
    validate_no_future_leakage: bool = True
    stop_on_critical_violation: bool = False
    max_decision_points: int | None = None

    # Metadata
    name: str = ""
    version: str = ""


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    # Traces
    traces: TraceCollection

    # Extraction stats
    extraction_result: ExtractionResult

    # Summary stats
    total_decisions: int = 0
    total_violations: int = 0
    critical_violations: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_ms: float = 0.0

    # Config used
    config_name: str = ""

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_decisions": self.total_decisions,
            "total_violations": self.total_violations,
            "critical_violations": self.critical_violations,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "config_name": self.config_name,
            "trace_summary": self.traces.summary(),
            "extraction_stats": {
                "cases_processed": self.extraction_result.cases_processed,
                "events_processed": self.extraction_result.events_processed,
                "rules_matched": self.extraction_result.rules_matched,
            },
            "errors": self.errors,
        }


class BenchmarkRunner:
    """
    Main runner for benchmark execution.

    Implements the golden path:
    events → decision points → policy decisions → metrics
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        constraint_checker: Callable[[PolicyResult, DecisionPoint, dict[str, Any]], list[ConstraintViolation]] | None = None,
    ) -> None:
        self.config = config
        self.constraint_checker = constraint_checker or self._default_constraint_checker

    def run(
        self,
        event_log: EventLog,
        policy: Policy,
        baseline: Policy | None = None,
    ) -> BenchmarkResult:
        """
        Run the benchmark.

        Args:
            event_log: Historical event log to replay
            policy: Policy to evaluate
            baseline: Optional baseline policy for comparison

        Returns:
            BenchmarkResult with traces and metrics
        """
        start_time = time.time()
        started_at = datetime.utcnow()
        errors: list[str] = []

        # Step 1: Extract decision points
        extraction_result = extract_decision_points(
            event_log=event_log,
            rules=self.config.decision_rules,
        )

        if extraction_result.errors:
            errors.extend(extraction_result.errors)

        decision_points = extraction_result.decision_points

        # Apply limit if configured
        if self.config.max_decision_points:
            decision_points = decision_points[: self.config.max_decision_points]

        # Step 2: Create timeline and replay context
        timeline = Timeline.from_decision_points(decision_points)
        replay_context = ReplayContext(timeline, event_log)

        # Step 3: Execute policy at each decision point
        traces = TraceCollection()
        total_violations = 0
        critical_violations = 0

        for dp in timeline.decision_points_only():
            # Validate no future leakage if configured
            if self.config.validate_no_future_leakage:
                warnings = replay_context.validate_no_future_leakage(dp)
                if warnings:
                    errors.extend(warnings)

            # Execute policy
            trace = self._execute_policy(policy, dp)
            traces.append(trace)

            # Count violations
            total_violations += len(trace.violations)
            critical_violations += sum(
                1 for v in trace.violations if v.severity == "critical"
            )

            # Stop on critical violation if configured
            if self.config.stop_on_critical_violation and trace.violations:
                if any(v.severity == "critical" for v in trace.violations):
                    errors.append(
                        f"Stopped at decision {dp.decision_id} due to critical violation"
                    )
                    break

        # Step 4: Optionally run baseline for comparison
        baseline_traces = None
        if baseline:
            baseline_traces = TraceCollection()
            for dp in timeline.decision_points_only():
                trace = self._execute_policy(baseline, dp)
                baseline_traces.append(trace)

        # Build result
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        return BenchmarkResult(
            traces=traces,
            extraction_result=extraction_result,
            total_decisions=len(traces),
            total_violations=total_violations,
            critical_violations=critical_violations,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            config_name=self.config.name,
            errors=errors,
        )

    def _execute_policy(
        self,
        policy: Policy,
        decision_point: DecisionPoint,
    ) -> ExecutionTrace:
        """Execute a policy at a decision point and create a trace."""
        start_time = time.time()

        # Get policy decision
        result = policy.decide(
            snapshot=decision_point.snapshot,
            allowed_actions=decision_point.allowed_actions,
        )

        execution_time_ms = (time.time() - start_time) * 1000

        # Check constraints
        violations = self.constraint_checker(result, decision_point, {})

        # Check if action matches historical
        matched_historical = False
        historical_action_type = None
        if decision_point.historical_action:
            historical_action_type = decision_point.historical_action.action_type
            matched_historical = (
                result.action.action_type == historical_action_type
            )

        return ExecutionTrace(
            decision_point=decision_point,
            result=result,
            executed_at=datetime.utcnow(),
            execution_time_ms=execution_time_ms,
            policy_name=policy.name,
            violations=violations,
            matched_historical=matched_historical,
            historical_action_type=historical_action_type,
        )

    def _default_constraint_checker(
        self,
        result: PolicyResult,
        decision_point: DecisionPoint,
        context: dict[str, Any],
    ) -> list[ConstraintViolation]:
        """Default constraint checker using config constraints."""
        violations = []

        for constraint in self.config.constraints:
            if self._check_constraint_violated(constraint, result, decision_point):
                violations.append(
                    ConstraintViolation(
                        constraint_name=constraint.name,
                        severity=constraint.severity,
                        message=constraint.message or f"Constraint {constraint.name} violated",
                        action=result.action,
                    )
                )

        return violations

    def _check_constraint_violated(
        self,
        constraint: Constraint,
        result: PolicyResult,
        decision_point: DecisionPoint,
    ) -> bool:
        """Check if a constraint is violated."""
        if not constraint.rule:
            return False

        # Build evaluation context
        context = {
            "action": result.action,
            "action_type": result.action.action_type,
            "action_params": result.action.params,
            "state": decision_point.snapshot.state,
            "confidence": result.confidence,
            "True": True,
            "False": False,
            "None": None,
        }

        # Add state fields
        context.update(decision_point.snapshot.state)
        context.update(decision_point.snapshot.features)

        try:
            # Constraint rule should return True if VIOLATED
            violated = eval(constraint.rule, {"__builtins__": {}}, context)
            return bool(violated)
        except Exception:
            return False


def run_benchmark(
    event_log: EventLog,
    decision_rules: list[DecisionPointRule],
    policy: Policy,
    constraints: list[Constraint] | None = None,
    baseline: Policy | None = None,
) -> BenchmarkResult:
    """
    Convenience function to run a benchmark.

    This is the main entry point for simple benchmark runs.
    """
    config = BenchmarkConfig(
        decision_rules=decision_rules,
        constraints=constraints or [],
    )

    runner = BenchmarkRunner(config)
    return runner.run(event_log, policy, baseline)
