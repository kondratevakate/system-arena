"""Lesson store for persisting and loading learned lessons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.feedback.types import LessonLearned


class LessonStore:
    """Store and retrieve lessons learned from analysis feedback.

    Lessons are stored in JSON format for git-friendliness and
    version control compatibility.
    """

    def __init__(self, lessons: list[LessonLearned] | None = None):
        self._lessons: dict[str, LessonLearned] = {}
        if lessons:
            for lesson in lessons:
                self._lessons[lesson.id] = lesson

    def add(self, lesson: LessonLearned) -> None:
        """Add a new lesson to the store."""
        self._lessons[lesson.id] = lesson

    def get(self, lesson_id: str) -> LessonLearned | None:
        """Get a lesson by ID."""
        return self._lessons.get(lesson_id)

    def remove(self, lesson_id: str) -> bool:
        """Remove a lesson by ID. Returns True if removed."""
        if lesson_id in self._lessons:
            del self._lessons[lesson_id]
            return True
        return False

    def get_for_metric(self, metric: str) -> list[LessonLearned]:
        """Get all lessons that apply to a specific metric."""
        return [
            lesson for lesson in self._lessons.values()
            if metric in lesson.applies_to or "all_metrics" in lesson.applies_to
        ]

    def get_by_pattern(self, pattern: str) -> list[LessonLearned]:
        """Get all lessons with a specific pattern."""
        return [
            lesson for lesson in self._lessons.values()
            if lesson.pattern == pattern
        ]

    def all(self) -> list[LessonLearned]:
        """Get all lessons."""
        return list(self._lessons.values())

    def __len__(self) -> int:
        return len(self._lessons)

    def __iter__(self):
        return iter(self._lessons.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert store to dictionary for serialization."""
        return {
            "lessons": [lesson.to_dict() for lesson in self._lessons.values()]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LessonStore:
        """Create store from dictionary."""
        lessons = [
            LessonLearned.from_dict(lesson_data)
            for lesson_data in data.get("lessons", [])
        ]
        return cls(lessons)

    def save(self, path: str | Path) -> None:
        """Save lessons to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> LessonStore:
        """Load lessons from JSON file."""
        path = Path(path)

        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def load_or_create(cls, path: str | Path) -> LessonStore:
        """Load lessons from file or create empty store if file doesn't exist."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls()
