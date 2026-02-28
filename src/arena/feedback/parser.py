"""Parser for converting GitHub issues to lessons."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from arena.feedback.types import (
    DataSchema,
    FeedbackIssue,
    LessonExample,
    LessonLearned,
    SystemContext,
)


# Mapping from issue form values to internal values
PATTERN_MAPPING = {
    "Zero denominator (0/0)": "zero_denominator",
    "Small sample size (n < 30)": "small_sample",
    "Missing event types": "missing_event_type",
    "Data quality issue": "data_quality",
    "Logic error": "logic_error",
    "Edge case not handled": "edge_case",
    "Other": "other",
}

METRIC_MAPPING = {
    "conversion_rate": "conversion_rate",
    "historical_match_rate": "historical_match_rate",
    "positive_outcome_rate": "positive_outcome_rate",
    "action_entropy": "action_entropy",
    "segment_comparison": "segment_comparison",
    "funnel_analysis": "funnel_analysis",
    "Other": "other",
}

# Default check expressions for common patterns
PATTERN_CHECKS = {
    "zero_denominator": "sample_size == 0",
    "small_sample": "sample_size < 30",
    "missing_event_type": "event_count == 0",
}

# Default warning messages for common patterns
PATTERN_WARNINGS = {
    "zero_denominator": "Cannot calculate rate: empty segment (sample_size=0)",
    "small_sample": "Low confidence: sample size {n} < 30",
    "missing_event_type": "Event type not found - check if logged",
}


def parse_issue_body(body: str) -> dict[str, Any]:
    """Parse a GitHub issue body into structured data.

    This handles both freeform markdown and structured form responses.
    """
    result: dict[str, Any] = {}

    # Try to extract form fields
    # GitHub forms produce markdown like:
    # ### Field Label
    # value

    sections = re.split(r"###\s+", body)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split("\n", 1)
        if len(lines) < 2:
            continue

        label = lines[0].strip().lower()
        value = lines[1].strip()

        # Map common field labels
        if "system type" in label:
            result["system_type"] = value
        elif "domain" in label:
            result["domain"] = value
        elif "user count" in label:
            result["user_count"] = _parse_number(value)
        elif "event count" in label:
            result["event_count"] = _parse_number(value)
        elif "metric" in label:
            result["metric"] = METRIC_MAPPING.get(value, value)
        elif "pattern" in label or "caused the error" in label:
            result["pattern"] = PATTERN_MAPPING.get(value, value)
        elif "reported" in label or "incorrect" in label:
            result["wrong_interpretation"] = value
        elif "correct" in label or "interpretation" in label:
            result["correct_interpretation"] = value
        elif "root cause" in label:
            result["root_cause"] = value
        elif "rule" in label or "prevent" in label:
            result["rule"] = value
        elif "event" in label and "context" in label:
            result["events_context"] = value

    return result


def _parse_number(value: str) -> int:
    """Parse a number from a string like '~700' or '2000'."""
    # Remove common prefixes
    value = value.replace("~", "").replace(",", "").strip()

    # Try to extract first number
    match = re.search(r"\d+", value)
    if match:
        return int(match.group())
    return 0


def parse_issue_to_feedback(
    issue_url: str,
    issue_body: str,
) -> FeedbackIssue:
    """Parse a GitHub issue into a FeedbackIssue object."""
    data = parse_issue_body(issue_body)

    system_context = None
    if data.get("system_type") or data.get("domain"):
        system_context = SystemContext(
            system_type=data.get("system_type", ""),
            domain=data.get("domain", ""),
            user_count=data.get("user_count", 0),
            event_count=data.get("event_count", 0),
        )

    return FeedbackIssue(
        url=issue_url,
        metric=data.get("metric", ""),
        pattern=data.get("pattern", ""),
        wrong_interpretation=data.get("wrong_interpretation", ""),
        correct_interpretation=data.get("correct_interpretation", ""),
        root_cause=data.get("root_cause", ""),
        rule=data.get("rule", ""),
        system_context=system_context,
        status="pending" if data.get("metric") else "needs_info",
    )


def feedback_to_lesson(
    feedback: FeedbackIssue,
    lesson_id: str | None = None,
) -> LessonLearned:
    """Convert a FeedbackIssue to a LessonLearned.

    Args:
        feedback: The parsed feedback issue
        lesson_id: Optional lesson ID; auto-generated if not provided

    Returns:
        A LessonLearned object
    """
    if lesson_id is None:
        lesson_id = f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine what metrics this applies to
    applies_to = [feedback.metric] if feedback.metric else []

    # Get default check and warning for the pattern
    check = PATTERN_CHECKS.get(feedback.pattern, "")
    warning = PATTERN_WARNINGS.get(feedback.pattern, "")

    # Create example from the feedback
    example_context = {}
    if feedback.system_context:
        example_context = {
            "domain": feedback.system_context.domain,
            "users": feedback.system_context.user_count,
        }

    example = LessonExample(
        wrong=feedback.wrong_interpretation,
        correct=feedback.correct_interpretation,
        context=example_context,
    )

    return LessonLearned(
        id=lesson_id,
        pattern=feedback.pattern,
        applies_to=applies_to,
        rule=feedback.rule,
        check=check,
        warning=warning if warning else feedback.rule,
        source_issue=feedback.url,
        added_at=datetime.now(),
        examples=[example] if feedback.wrong_interpretation else [],
    )


def generate_missing_info_comment(feedback: FeedbackIssue) -> str:
    """Generate a comment asking for missing information."""
    missing = feedback.get_missing_fields()

    if not missing:
        return ""

    comment_parts = [
        "Thanks for the feedback! To learn from this case, I need more context:\n"
    ]

    if "system_type" in missing or "domain" in missing or "system_context" in missing:
        comment_parts.append("### System Context (missing)")
        if "system_type" in missing or "system_context" in missing:
            comment_parts.append("- [ ] What type of system? (Telegram bot, web app, mobile app)")
        if "domain" in missing:
            comment_parts.append("- [ ] What domain? (Travel, e-commerce, education)")
        comment_parts.append("")

    if "user_count" in missing or "event_count" in missing:
        comment_parts.append("### Data Context (missing)")
        if "user_count" in missing:
            comment_parts.append("- [ ] How many users in total?")
        comment_parts.append("")

    if "metric" in missing or "wrong_interpretation" in missing:
        comment_parts.append("### Analysis Context (missing)")
        if "metric" in missing:
            comment_parts.append("- [ ] What metric was being calculated?")
        if "wrong_interpretation" in missing:
            comment_parts.append("- [ ] What did arena report incorrectly?")
        comment_parts.append("")

    if "correct_interpretation" in missing or "rule" in missing:
        comment_parts.append("### Correction (missing)")
        if "correct_interpretation" in missing:
            comment_parts.append("- [ ] What should have been reported instead?")
        if "rule" in missing:
            comment_parts.append("- [ ] What rule should prevent this error?")
        comment_parts.append("")

    comment_parts.append("Please fill in the missing sections so I can add this as a lesson.")

    return "\n".join(comment_parts)
