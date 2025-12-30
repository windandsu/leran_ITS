"""管理学生画像的轻量级存储。"""
from __future__ import annotations

from typing import Dict

from its.data_models import StudentProfile


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
