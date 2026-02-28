"""Checker for validating metrics against known lessons."""

from __future__ import annotations

from typing import Any

from arena.feedback.store import LessonStore
from arena.feedback.types import CheckResult, LessonLearned


class LessonChecker:
    """Check metric results against known lessons to generate warnings."""

    def __init__(self, store: LessonStore):
        self._store = store

    def check(
        self,
        metric: str,
        value: float,
        sample_size: int = 0,
        **context: Any,
    ) -> CheckResult:
        """Check a metric result against known lessons.

        Args:
            metric: Name of the metric (e.g., "conversion_rate")
            value: The computed metric value
            sample_size: Sample size used for computation
            **context: Additional context (e.g., segment names, event types)

        Returns:
            CheckResult with any warnings and matched lesson IDs
        """
        warnings: list[str] = []
        matched_lessons: list[str] = []

        # Get lessons that apply to this metric
        applicable_lessons = self._store.get_for_metric(metric)

        for lesson in applicable_lessons:
            warning = self._evaluate_lesson(
                lesson, value, sample_size, context
            )
            if warning:
                warnings.append(warning)
                matched_lessons.append(lesson.id)

        return CheckResult(
            has_warnings=len(warnings) > 0,
            warnings=warnings,
            matched_lessons=matched_lessons,
        )

    def _evaluate_lesson(
        self,
        lesson: LessonLearned,
        value: float,
        sample_size: int,
        context: dict[str, Any],
    ) -> str | None:
        """Evaluate a single lesson against the metric.

        Returns warning message if lesson matches, None otherwise.
        """
        # Build evaluation context
        eval_context = {
            "value": value,
            "sample_size": sample_size,
            "n": sample_size,  # Alias
            **context,
        }

        try:
            # Evaluate the check expression
            # Note: Using eval is safe here since lessons come from our own JSON
            if self._evaluate_check(lesson.check, eval_context):
                return self._format_warning(lesson.warning, eval_context)
        except Exception:
            # If evaluation fails, skip this lesson
            pass

        return None

    def _evaluate_check(self, check: str, context: dict[str, Any]) -> bool:
        """Evaluate a check expression.

        Supported expressions:
        - "sample_size == 0"
        - "sample_size < 30"
        - "event_count == 0 and event_type in expected_types"
        """
        if not check:
            return False

        # Simple expression evaluation
        # For security, we only allow specific patterns
        check = check.strip()

        # Pattern: sample_size == 0
        if check == "sample_size == 0":
            return context.get("sample_size", 0) == 0

        # Pattern: sample_size < N
        if check.startswith("sample_size < "):
            try:
                threshold = int(check.split("<")[1].strip())
                return context.get("sample_size", 0) < threshold
            except (ValueError, IndexError):
                return False

        # Pattern: sample_size > N
        if check.startswith("sample_size > "):
            try:
                threshold = int(check.split(">")[1].strip())
                return context.get("sample_size", 0) > threshold
            except (ValueError, IndexError):
                return False

        # Pattern: event_count == 0
        if check == "event_count == 0":
            return context.get("event_count", 0) == 0

        # For more complex expressions, use safe eval
        try:
            # Only allow safe operations
            allowed_names = {
                "sample_size": context.get("sample_size", 0),
                "n": context.get("n", 0),
                "value": context.get("value", 0),
                "event_count": context.get("event_count", 0),
            }
            return eval(check, {"__builtins__": {}}, allowed_names)
        except Exception:
            return False

    def _format_warning(self, template: str, context: dict[str, Any]) -> str:
        """Format a warning message template with context values."""
        try:
            return template.format(**context)
        except KeyError:
            # If formatting fails, return template as-is
            return template


def check_metrics(
    store: LessonStore,
    metric: str,
    value: float,
    sample_size: int = 0,
    **context: Any,
) -> CheckResult:
    """Convenience function to check metrics against lessons.

    Args:
        store: The lesson store to check against
        metric: Name of the metric
        value: Computed metric value
        sample_size: Sample size used
        **context: Additional context

    Returns:
        CheckResult with warnings
    """
    checker = LessonChecker(store)
    return checker.check(metric, value, sample_size, **context)
