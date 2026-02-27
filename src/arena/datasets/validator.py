"""Schema validation for datasets and manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.datasets.manifest import WorkflowManifest


@dataclass
class ValidationError:
    """A validation error."""

    field: str
    message: str
    severity: str = "error"  # "error", "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, field: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field, message=message, severity="error"))
        self.valid = False

    def add_warning(self, field: str, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(field=field, message=message, severity="warning"))

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [{"field": e.field, "message": e.message} for e in self.errors],
            "warnings": [{"field": w.field, "message": w.message} for w in self.warnings],
        }

    def summary(self) -> str:
        """Get a text summary."""
        if self.valid and not self.warnings:
            return "Validation passed."

        lines = []
        if not self.valid:
            lines.append(f"Validation FAILED with {self.error_count} error(s).")
        else:
            lines.append("Validation passed with warnings.")

        if self.warnings:
            lines.append(f"{self.warning_count} warning(s).")

        for error in self.errors:
            lines.append(f"  [ERROR] {error.field}: {error.message}")

        for warning in self.warnings:
            lines.append(f"  [WARN] {warning.field}: {warning.message}")

        return "\n".join(lines)


class ManifestValidator:
    """Validates workflow manifests."""

    def validate(self, manifest: WorkflowManifest) -> ValidationResult:
        """Validate a workflow manifest."""
        result = ValidationResult(valid=True)

        # Check required fields
        if not manifest.name:
            result.add_error("name", "Manifest name is required")

        # Check decision points
        if not manifest.decision_points:
            result.add_warning("decision_points", "No decision points defined")
        else:
            for i, dp in enumerate(manifest.decision_points):
                self._validate_decision_point(dp, i, result)

        # Check allowed actions
        if not manifest.allowed_actions:
            result.add_warning("allowed_actions", "No allowed actions defined")

        # Check outcomes
        if not any([
            manifest.outcomes.positive,
            manifest.outcomes.negative,
            manifest.outcomes.neutral,
        ]):
            result.add_warning("outcomes", "No outcomes defined")

        # Check constraints
        for i, constraint in enumerate(manifest.constraints):
            self._validate_constraint(constraint, i, result)

        return result

    def _validate_decision_point(
        self,
        dp: Any,
        index: int,
        result: ValidationResult,
    ) -> None:
        """Validate a decision point configuration."""
        prefix = f"decision_points[{index}]"

        if not dp.name:
            result.add_error(f"{prefix}.name", "Decision point name is required")

        if not dp.trigger:
            result.add_error(f"{prefix}.trigger", "Decision point trigger is required")

        # Validate condition syntax (basic check)
        if dp.condition:
            try:
                compile(dp.condition, "<condition>", "eval")
            except SyntaxError as e:
                result.add_error(f"{prefix}.condition", f"Invalid Python expression: {e}")

    def _validate_constraint(
        self,
        constraint: Any,
        index: int,
        result: ValidationResult,
    ) -> None:
        """Validate a constraint configuration."""
        prefix = f"constraints[{index}]"

        if not constraint.name:
            result.add_error(f"{prefix}.name", "Constraint name is required")

        if not constraint.rule:
            result.add_error(f"{prefix}.rule", "Constraint rule is required")
        else:
            # Validate rule syntax
            try:
                compile(constraint.rule, "<rule>", "eval")
            except SyntaxError as e:
                result.add_error(f"{prefix}.rule", f"Invalid Python expression: {e}")

        if constraint.severity not in ("warning", "violation", "critical"):
            result.add_warning(
                f"{prefix}.severity",
                f"Unknown severity '{constraint.severity}', expected warning/violation/critical",
            )


class EventLogValidator:
    """Validates event log data."""

    def __init__(self, manifest: WorkflowManifest | None = None) -> None:
        self.manifest = manifest
        self.id_field = manifest.case.id_field if manifest else "case_id"
        self.timestamp_field = manifest.case.timestamp_field if manifest else "timestamp"

    def validate_record(self, record: dict[str, Any], index: int) -> ValidationResult:
        """Validate a single event record."""
        result = ValidationResult(valid=True)
        prefix = f"record[{index}]"

        # Check required fields
        if self.id_field not in record or record[self.id_field] is None:
            result.add_error(f"{prefix}.{self.id_field}", "Case ID is required")

        if self.timestamp_field not in record or record[self.timestamp_field] is None:
            result.add_error(f"{prefix}.{self.timestamp_field}", "Timestamp is required")

        # Validate timestamp format
        if self.timestamp_field in record and record[self.timestamp_field]:
            try:
                from arena.datasets.loader import _parse_timestamp
                _parse_timestamp(record[self.timestamp_field])
            except ValueError as e:
                result.add_error(f"{prefix}.{self.timestamp_field}", f"Invalid timestamp: {e}")

        return result

    def validate_log(self, records: list[dict[str, Any]]) -> ValidationResult:
        """Validate a full event log."""
        result = ValidationResult(valid=True)

        if not records:
            result.add_error("records", "Event log is empty")
            return result

        # Check sample of records (for performance)
        sample_size = min(100, len(records))
        for i in range(sample_size):
            record_result = self.validate_record(records[i], i)
            result.merge(record_result)

        # Check temporal ordering
        self._check_temporal_ordering(records, result)

        # Check case completeness
        self._check_case_completeness(records, result)

        return result

    def _check_temporal_ordering(
        self,
        records: list[dict[str, Any]],
        result: ValidationResult,
    ) -> None:
        """Check that timestamps are reasonable."""
        from arena.datasets.loader import _parse_timestamp

        try:
            timestamps = []
            for record in records[:100]:  # Sample
                if self.timestamp_field in record:
                    timestamps.append(_parse_timestamp(record[self.timestamp_field]))

            if timestamps:
                min_ts = min(timestamps)
                max_ts = max(timestamps)
                span = max_ts - min_ts

                if span.days > 365 * 10:
                    result.add_warning(
                        "temporal_range",
                        f"Event log spans {span.days} days; verify this is correct",
                    )
        except Exception:
            pass

    def _check_case_completeness(
        self,
        records: list[dict[str, Any]],
        result: ValidationResult,
    ) -> None:
        """Check that cases have reasonable event counts."""
        case_counts: dict[str, int] = {}

        for record in records:
            case_id = str(record.get(self.id_field, ""))
            case_counts[case_id] = case_counts.get(case_id, 0) + 1

        if case_counts:
            counts = list(case_counts.values())
            avg_events = sum(counts) / len(counts)

            if avg_events < 2:
                result.add_warning(
                    "case_completeness",
                    f"Average events per case is {avg_events:.1f}; most cases have only 1 event",
                )

            # Check for single-event cases
            single_event_cases = sum(1 for c in counts if c == 1)
            if single_event_cases > len(counts) * 0.5:
                result.add_warning(
                    "single_event_cases",
                    f"{single_event_cases}/{len(counts)} cases have only 1 event",
                )


def validate_manifest(manifest: WorkflowManifest) -> ValidationResult:
    """Convenience function to validate a manifest."""
    validator = ManifestValidator()
    return validator.validate(manifest)


def validate_directory(
    directory: str | Path,
    manifest_name: str = "manifest.yaml",
    data_pattern: str = "*.parquet",
) -> ValidationResult:
    """
    Validate a workflow directory with manifest and data files.

    Args:
        directory: Path to the workflow directory
        manifest_name: Name of the manifest file
        data_pattern: Glob pattern for data files

    Returns:
        Combined validation result
    """
    from arena.datasets.loader import load_event_log
    from arena.datasets.manifest import load_manifest

    directory = Path(directory)
    result = ValidationResult(valid=True)

    # Check directory exists
    if not directory.exists():
        result.add_error("directory", f"Directory not found: {directory}")
        return result

    # Load and validate manifest
    manifest_path = directory / manifest_name

    if not manifest_path.exists():
        result.add_error("manifest", f"Manifest not found: {manifest_path}")
        return result

    try:
        manifest = load_manifest(manifest_path)
        manifest_result = validate_manifest(manifest)
        result.merge(manifest_result)
    except Exception as e:
        result.add_error("manifest", f"Failed to load manifest: {e}")
        return result

    # Find and validate data files
    data_files = list(directory.glob(data_pattern))

    if not data_files:
        result.add_error("data", f"No data files matching {data_pattern}")
        return result

    # Validate each data file
    event_validator = EventLogValidator(manifest)

    for data_file in data_files:
        try:
            event_log = load_event_log(data_file, manifest)
            records = event_log.to_dicts()
            log_result = event_validator.validate_log(records)
            result.merge(log_result)
        except Exception as e:
            result.add_error(f"data.{data_file.name}", f"Failed to load: {e}")

    return result
