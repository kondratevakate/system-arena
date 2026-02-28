"""Tests for the arena.feedback module."""

import json
import tempfile
from pathlib import Path

import pytest

from arena.feedback import (
    CheckResult,
    LessonChecker,
    LessonExample,
    LessonLearned,
    LessonStore,
    SystemContext,
    check_metrics,
    feedback_to_lesson,
    generate_missing_info_comment,
    parse_issue_to_feedback,
)


class TestLessonStore:
    """Tests for LessonStore."""

    def test_empty_store(self):
        store = LessonStore()
        assert len(store) == 0
        assert store.all() == []

    def test_add_lesson(self):
        store = LessonStore()
        lesson = LessonLearned(
            id="test_001",
            pattern="zero_denominator",
            applies_to=["conversion_rate"],
            rule="Test rule",
            check="sample_size == 0",
            warning="Test warning",
        )
        store.add(lesson)
        assert len(store) == 1
        assert store.get("test_001") == lesson

    def test_get_for_metric(self):
        store = LessonStore()
        lesson1 = LessonLearned(
            id="test_001",
            pattern="zero_denominator",
            applies_to=["conversion_rate"],
            rule="Test rule 1",
            check="sample_size == 0",
            warning="Warning 1",
        )
        lesson2 = LessonLearned(
            id="test_002",
            pattern="small_sample",
            applies_to=["all_metrics"],
            rule="Test rule 2",
            check="sample_size < 30",
            warning="Warning 2",
        )
        store.add(lesson1)
        store.add(lesson2)

        # Should get both: one specific and one for all_metrics
        results = store.get_for_metric("conversion_rate")
        assert len(results) == 2

        # Should only get the all_metrics one
        results = store.get_for_metric("other_metric")
        assert len(results) == 1
        assert results[0].id == "test_002"

    def test_save_and_load(self):
        store = LessonStore()
        lesson = LessonLearned(
            id="test_001",
            pattern="zero_denominator",
            applies_to=["conversion_rate"],
            rule="Test rule",
            check="sample_size == 0",
            warning="Test warning",
            examples=[
                LessonExample(
                    wrong="0/0 = 0%",
                    correct="Insufficient data",
                    context={"users": 100},
                )
            ],
        )
        store.add(lesson)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = Path(f.name)

        try:
            store.save(path)
            loaded = LessonStore.load(path)

            assert len(loaded) == 1
            loaded_lesson = loaded.get("test_001")
            assert loaded_lesson is not None
            assert loaded_lesson.pattern == "zero_denominator"
            assert len(loaded_lesson.examples) == 1
        finally:
            path.unlink()


class TestLessonChecker:
    """Tests for LessonChecker."""

    def test_check_zero_denominator(self):
        store = LessonStore()
        lesson = LessonLearned(
            id="test_001",
            pattern="zero_denominator",
            applies_to=["conversion_rate"],
            rule="Check for zero denominator",
            check="sample_size == 0",
            warning="Cannot calculate: empty segment",
        )
        store.add(lesson)

        # Should trigger warning
        result = check_metrics(store, "conversion_rate", 0.0, sample_size=0)
        assert result.has_warnings
        assert "Cannot calculate: empty segment" in result.warnings
        assert "test_001" in result.matched_lessons

        # Should not trigger warning
        result = check_metrics(store, "conversion_rate", 0.5, sample_size=100)
        assert not result.has_warnings

    def test_check_small_sample(self):
        store = LessonStore()
        lesson = LessonLearned(
            id="test_002",
            pattern="small_sample",
            applies_to=["all_metrics"],
            rule="Warn on small samples",
            check="sample_size < 30",
            warning="Low confidence: sample size {n} < 30",
        )
        store.add(lesson)

        # Should trigger warning
        result = check_metrics(store, "conversion_rate", 1.0, sample_size=5)
        assert result.has_warnings
        assert "Low confidence: sample size 5 < 30" in result.warnings

        # Should not trigger warning
        result = check_metrics(store, "conversion_rate", 0.5, sample_size=100)
        assert not result.has_warnings


class TestIssueParser:
    """Tests for issue parsing."""

    def test_parse_issue_body(self):
        issue_body = """
### System Type
Telegram bot

### Domain
Travel

### Approximate User Count
~700

### Approximate Event Count
~2000

### Which metric was affected?
conversion_rate

### What caused the error?
Zero denominator (0/0)

### What arena reported (incorrect)
Feature does not help conversion (0.0% vs 5.0%)

### What the correct interpretation is
Insufficient data to evaluate feature

### Root Cause
Arena assumed 0/0 = 0%

### Rule to Prevent This
Check if denominator is 0 before calculating rate
"""
        feedback = parse_issue_to_feedback(
            "https://github.com/test/repo/issues/1",
            issue_body,
        )

        assert feedback.metric == "conversion_rate"
        assert feedback.pattern == "zero_denominator"
        assert "0.0%" in feedback.wrong_interpretation
        assert "Insufficient data" in feedback.correct_interpretation
        assert feedback.system_context is not None
        assert feedback.system_context.system_type == "Telegram bot"
        assert feedback.system_context.domain == "Travel"
        assert feedback.system_context.user_count == 700

    def test_feedback_to_lesson(self):
        feedback = parse_issue_to_feedback(
            "https://github.com/test/repo/issues/1",
            """
### System Type
Web application

### Domain
E-commerce

### Approximate User Count
500

### Which metric was affected?
conversion_rate

### What caused the error?
Zero denominator (0/0)

### What arena reported (incorrect)
Segment has 0% conversion

### What the correct interpretation is
Insufficient data

### Root Cause
Division by zero

### Rule to Prevent This
Check sample size before division
""",
        )

        lesson = feedback_to_lesson(feedback, lesson_id="test_lesson")

        assert lesson.id == "test_lesson"
        assert lesson.pattern == "zero_denominator"
        assert "conversion_rate" in lesson.applies_to
        assert lesson.rule == "Check sample size before division"

    def test_generate_missing_info_comment(self):
        feedback = parse_issue_to_feedback(
            "https://github.com/test/repo/issues/1",
            """
### Which metric was affected?
conversion_rate
""",
        )

        comment = generate_missing_info_comment(feedback)

        assert "System Context (missing)" in comment
        assert "What type of system?" in comment


class TestSystemContext:
    """Tests for SystemContext."""

    def test_to_dict_and_from_dict(self):
        context = SystemContext(
            system_type="telegram_bot",
            domain="travel",
            business_model="freemium",
            user_count=700,
            event_count=2000,
            time_span_days=180,
            conversion_rate=0.05,
        )

        data = context.to_dict()
        restored = SystemContext.from_dict(data)

        assert restored.system_type == "telegram_bot"
        assert restored.domain == "travel"
        assert restored.user_count == 700
