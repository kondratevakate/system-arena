"""Report generation for benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from arena.eval.comparator import ComparisonResult
from arena.eval.metrics import MetricsSummary, compute_metrics
from arena.replay.runner import BenchmarkResult
from arena.replay.trace import TraceCollection


@dataclass
class ReportSection:
    """A section in the benchmark report."""

    title: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    title: str
    generated_at: datetime
    sections: list[ReportSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def to_text(self) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 70,
            self.title.center(70),
            "=" * 70,
            f"Generated: {self.generated_at.isoformat()}",
            "",
        ]

        for section in self.sections:
            lines.extend([
                "-" * 70,
                section.title,
                "-" * 70,
                section.content,
                "",
            ])

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {self.generated_at.isoformat()}*",
            "",
        ]

        for section in self.sections:
            lines.extend([
                f"## {section.title}",
                "",
                section.content,
                "",
            ])

        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(
            {
                "title": self.title,
                "generated_at": self.generated_at.isoformat(),
                "sections": [
                    {
                        "title": s.title,
                        "content": s.content,
                        "data": s.data,
                    }
                    for s in self.sections
                ],
                "metadata": self.metadata,
            },
            indent=2,
            default=str,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "sections": [
                {"title": s.title, "data": s.data}
                for s in self.sections
            ],
            "metadata": self.metadata,
        }


class ReportGenerator:
    """Generates benchmark reports from results."""

    def __init__(self, title: str = "Benchmark Report") -> None:
        self.title = title

    def generate(
        self,
        result: BenchmarkResult,
        comparison: ComparisonResult | None = None,
    ) -> BenchmarkReport:
        """
        Generate a complete benchmark report.

        Args:
            result: Benchmark result to report on
            comparison: Optional policy comparison result

        Returns:
            BenchmarkReport ready for output
        """
        report = BenchmarkReport(
            title=self.title,
            generated_at=datetime.utcnow(),
            metadata={
                "config_name": result.config_name,
                "total_decisions": result.total_decisions,
            },
        )

        # Add summary section
        report.add_section(self._generate_summary_section(result))

        # Add metrics section
        metrics = compute_metrics(result.traces)
        report.add_section(self._generate_metrics_section(metrics))

        # Add extraction section
        report.add_section(self._generate_extraction_section(result))

        # Add violations section if there are violations
        if result.total_violations > 0:
            report.add_section(self._generate_violations_section(result))

        # Add comparison section if comparison provided
        if comparison:
            report.add_section(self._generate_comparison_section(comparison))

        # Add action distribution section
        report.add_section(self._generate_action_distribution_section(result.traces))

        return report

    def _generate_summary_section(self, result: BenchmarkResult) -> ReportSection:
        """Generate the summary section."""
        content = f"""
Total Decision Points: {result.total_decisions}
Cases Processed: {result.extraction_result.cases_processed}
Events Processed: {result.extraction_result.events_processed}

Execution Time: {result.duration_ms:.2f} ms
Violations: {result.total_violations} ({result.critical_violations} critical)

Errors: {len(result.errors)}
""".strip()

        return ReportSection(
            title="Summary",
            content=content,
            data=result.to_dict(),
        )

    def _generate_metrics_section(self, metrics: MetricsSummary) -> ReportSection:
        """Generate the metrics section."""
        lines = []

        for name, metric_result in metrics.metrics.items():
            lines.append(f"  {name}: {metric_result.value:.4f}")

        content = "\n".join(lines) if lines else "No metrics computed."

        return ReportSection(
            title="Metrics",
            content=content,
            data=metrics.to_dict(),
        )

    def _generate_extraction_section(self, result: BenchmarkResult) -> ReportSection:
        """Generate the extraction statistics section."""
        lines = ["Rules Matched:"]

        for rule_name, count in result.extraction_result.rules_matched.items():
            lines.append(f"  {rule_name}: {count}")

        content = "\n".join(lines)

        return ReportSection(
            title="Decision Point Extraction",
            content=content,
            data={"rules_matched": result.extraction_result.rules_matched},
        )

    def _generate_violations_section(self, result: BenchmarkResult) -> ReportSection:
        """Generate the violations section."""
        # Group violations by constraint
        violation_counts: dict[str, int] = {}

        for trace in result.traces:
            for violation in trace.violations:
                key = violation.constraint_name
                violation_counts[key] = violation_counts.get(key, 0) + 1

        lines = [f"Total Violations: {result.total_violations}"]
        lines.append(f"Critical: {result.critical_violations}")
        lines.append("")
        lines.append("By Constraint:")

        for constraint, count in sorted(
            violation_counts.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {constraint}: {count}")

        return ReportSection(
            title="Constraint Violations",
            content="\n".join(lines),
            data={"violation_counts": violation_counts},
        )

    def _generate_comparison_section(
        self, comparison: ComparisonResult
    ) -> ReportSection:
        """Generate the policy comparison section."""
        lines = [
            f"Policy: {comparison.policy_name}",
            f"Baseline: {comparison.baseline_name}",
            f"Agreement Rate: {comparison.agreement_rate:.1%}",
            "",
            "Metric Differences (policy - baseline):",
        ]

        for metric, diff in comparison.metric_diffs.items():
            uplift_pct = comparison.uplift.get(metric, 0.0) * 100
            sign = "+" if diff >= 0 else ""
            lines.append(f"  {metric}: {sign}{diff:.4f} ({sign}{uplift_pct:.1f}%)")

        return ReportSection(
            title="Policy Comparison",
            content="\n".join(lines),
            data=comparison.to_dict(),
        )

    def _generate_action_distribution_section(
        self, traces: TraceCollection
    ) -> ReportSection:
        """Generate the action distribution section."""
        dist = traces.action_distribution
        total = sum(dist.values())

        lines = []
        for action, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {action}: {count} ({pct:.1f}%)")

        return ReportSection(
            title="Action Distribution",
            content="\n".join(lines) if lines else "No actions recorded.",
            data={"distribution": dist},
        )


def generate_report(
    result: BenchmarkResult,
    comparison: ComparisonResult | None = None,
    title: str = "Benchmark Report",
) -> BenchmarkReport:
    """
    Convenience function to generate a benchmark report.

    Args:
        result: Benchmark result to report on
        comparison: Optional comparison result
        title: Report title

    Returns:
        BenchmarkReport ready for output
    """
    generator = ReportGenerator(title=title)
    return generator.generate(result, comparison)


def print_report(
    result: BenchmarkResult,
    comparison: ComparisonResult | None = None,
    format: str = "text",
) -> None:
    """
    Generate and print a benchmark report.

    Args:
        result: Benchmark result
        comparison: Optional comparison result
        format: Output format ("text", "markdown", "json")
    """
    report = generate_report(result, comparison)

    if format == "text":
        print(report.to_text())
    elif format == "markdown":
        print(report.to_markdown())
    elif format == "json":
        print(report.to_json())
    else:
        raise ValueError(f"Unknown format: {format}")
