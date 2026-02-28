"""Arena Feedback System.

This module provides a feedback loop for arena to learn from analysis mistakes.
Users can report errors via GitHub Issues, which are then parsed into lessons
that arena applies to future analyses.

Example usage:

    from arena.feedback import LessonStore, check_metrics

    # Load lessons from JSON
    store = LessonStore.load("lessons/edge_cases.json")

    # Check a metric against known lessons
    result = check_metrics(
        store,
        metric="conversion_rate",
        value=0.0,
        sample_size=0,
    )

    if result.has_warnings:
        print("Warnings:", result.warnings)
        # → ["Cannot calculate rate: empty segment (sample_size=0)"]
"""

from arena.feedback.checker import LessonChecker, check_metrics
from arena.feedback.parser import (
    feedback_to_lesson,
    generate_missing_info_comment,
    parse_issue_to_feedback,
)
from arena.feedback.store import LessonStore
from arena.feedback.types import (
    CheckResult,
    DataSchema,
    FeedbackIssue,
    LessonExample,
    LessonLearned,
    SystemContext,
)

__all__ = [
    # Types
    "CheckResult",
    "DataSchema",
    "FeedbackIssue",
    "LessonExample",
    "LessonLearned",
    "SystemContext",
    # Store
    "LessonStore",
    # Checker
    "LessonChecker",
    "check_metrics",
    # Parser
    "parse_issue_to_feedback",
    "feedback_to_lesson",
    "generate_missing_info_comment",
]
