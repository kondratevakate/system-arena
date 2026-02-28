"""Types for the Arena feedback system.

This module defines dataclasses for capturing analysis feedback,
lessons learned, and system context - all anonymously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SystemContext:
    """Anonymous context about the system being analyzed.

    This captures enough information to understand the analysis context
    without identifying the specific project or organization.
    """
    system_type: str  # "telegram_bot", "web_app", "mobile_app", etc.
    domain: str  # "travel", "e-commerce", "education", etc.
    business_model: str = ""  # "freemium", "subscription", "one-time"
    user_count: int = 0
    event_count: int = 0
    time_span_days: int = 0
    conversion_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_type": self.system_type,
            "domain": self.domain,
            "business_model": self.business_model,
            "user_count": self.user_count,
            "event_count": self.event_count,
            "time_span_days": self.time_span_days,
            "conversion_rate": self.conversion_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemContext:
        return cls(
            system_type=data.get("system_type", ""),
            domain=data.get("domain", ""),
            business_model=data.get("business_model", ""),
            user_count=data.get("user_count", 0),
            event_count=data.get("event_count", 0),
            time_span_days=data.get("time_span_days", 0),
            conversion_rate=data.get("conversion_rate", 0.0),
        )


@dataclass
class DataSchema:
    """What events exist in the system."""
    original_events: list[str] = field(default_factory=list)
    missing_events: list[str] = field(default_factory=list)
    added_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_events": self.original_events,
            "missing_events": self.missing_events,
            "added_events": self.added_events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSchema:
        return cls(
            original_events=data.get("original_events", []),
            missing_events=data.get("missing_events", []),
            added_events=data.get("added_events", []),
        )


@dataclass
class LessonExample:
    """A concrete example of an analysis error."""
    wrong: str  # What was incorrectly reported
    correct: str  # What should have been reported
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrong": self.wrong,
            "correct": self.correct,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LessonExample:
        return cls(
            wrong=data.get("wrong", ""),
            correct=data.get("correct", ""),
            context=data.get("context", {}),
        )


@dataclass
class LessonLearned:
    """A lesson learned from analysis feedback.

    This represents a pattern that arena should watch for
    and warn about in future analyses.
    """
    id: str
    pattern: str  # "zero_denominator", "small_sample", etc.
    applies_to: list[str]  # ["conversion_rate", "segment_comparison"]
    rule: str  # Human-readable rule
    check: str  # Condition expression, e.g., "sample_size == 0"
    warning: str  # Warning message template
    source_issue: str = ""  # GitHub issue URL
    added_at: datetime = field(default_factory=datetime.now)
    examples: list[LessonExample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "applies_to": self.applies_to,
            "rule": self.rule,
            "check": self.check,
            "warning": self.warning,
            "source_issue": self.source_issue,
            "added_at": self.added_at.isoformat(),
            "examples": [ex.to_dict() for ex in self.examples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LessonLearned:
        added_at = data.get("added_at")
        if isinstance(added_at, str):
            added_at = datetime.fromisoformat(added_at)
        elif added_at is None:
            added_at = datetime.now()

        return cls(
            id=data.get("id", ""),
            pattern=data.get("pattern", ""),
            applies_to=data.get("applies_to", []),
            rule=data.get("rule", ""),
            check=data.get("check", ""),
            warning=data.get("warning", ""),
            source_issue=data.get("source_issue", ""),
            added_at=added_at,
            examples=[
                LessonExample.from_dict(ex)
                for ex in data.get("examples", [])
            ],
        )


@dataclass
class FeedbackIssue:
    """Parsed feedback from a GitHub issue."""
    url: str
    metric: str
    pattern: str
    wrong_interpretation: str
    correct_interpretation: str
    root_cause: str
    rule: str
    system_context: SystemContext | None = None
    data_schema: DataSchema | None = None
    status: str = "pending"  # "pending", "parsed", "needs_info", "resolved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "metric": self.metric,
            "pattern": self.pattern,
            "wrong_interpretation": self.wrong_interpretation,
            "correct_interpretation": self.correct_interpretation,
            "root_cause": self.root_cause,
            "rule": self.rule,
            "system_context": self.system_context.to_dict() if self.system_context else None,
            "data_schema": self.data_schema.to_dict() if self.data_schema else None,
            "status": self.status,
        }

    def get_missing_fields(self) -> list[str]:
        """Check what required information is missing."""
        missing = []

        if not self.metric:
            missing.append("metric")
        if not self.pattern:
            missing.append("pattern")
        if not self.wrong_interpretation:
            missing.append("wrong_interpretation")
        if not self.correct_interpretation:
            missing.append("correct_interpretation")
        if not self.rule:
            missing.append("rule")

        # System context checks
        if not self.system_context:
            missing.append("system_context")
        elif not self.system_context.system_type:
            missing.append("system_type")
        elif not self.system_context.domain:
            missing.append("domain")
        elif self.system_context.user_count == 0:
            missing.append("user_count")

        return missing

    def is_complete(self) -> bool:
        """Check if the feedback has all required information."""
        return len(self.get_missing_fields()) == 0


@dataclass
class CheckResult:
    """Result of checking a metric against known lessons."""
    has_warnings: bool
    warnings: list[str] = field(default_factory=list)
    matched_lessons: list[str] = field(default_factory=list)  # Lesson IDs

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_warnings": self.has_warnings,
            "warnings": self.warnings,
            "matched_lessons": self.matched_lessons,
        }
