"""管理学生画像的轻量级存储。"""
from __future__ import annotations

from typing import Dict, List

from its.data_models import PlanItem, StudentProfile


class ProfileStoreTool:
    """简单的内存画像存储，未来可替换为数据库。"""

    def __init__(self) -> None:
        self._profiles: Dict[str, StudentProfile] = {}

    def get(self, student_id: str) -> StudentProfile:
        return self._profiles.setdefault(
            student_id, StudentProfile(student_id=student_id, goals=[], skills={}, preferences={})
        )

    def update(self, student_id: str, updates: dict) -> StudentProfile:
        profile = self.get(student_id)
        for key, value in updates.items():
            setattr(profile, key, value)
        return profile

    def append_history(self, student_id: str, record: str) -> None:
        profile = self.get(student_id)
        profile.history.append(record)

    def add_question(self, student_id: str, question: str, keep_last: int = 5) -> None:
        profile = self.get(student_id)
        profile.recent_questions.append(question)
        profile.recent_questions = profile.recent_questions[-keep_last:]

    def upsert_plan(self, student_id: str, plan_items: List[PlanItem]) -> List[PlanItem]:
        profile = self.get(student_id)
        profile.schedule = plan_items
        return profile.schedule

    def add_progress_note(self, student_id: str, note: str) -> None:
        profile = self.get(student_id)
        profile.progress_notes.append(note)
