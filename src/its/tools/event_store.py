"""学习事件日志存储。"""
from __future__ import annotations

from typing import Dict, List, Optional

from its.data_models import AgentTask, LearningEvent


class EventStoreTool:
    """记录并查询学习事件，可替换为消息队列或数据库。"""

    def __init__(self) -> None:
        self._events: List[LearningEvent] = []

    def log(self, task: AgentTask, detail: str, success: bool = True, metadata: Optional[Dict] = None) -> None:
        event = LearningEvent(
            student_id=metadata.get("student_id", "unknown") if metadata else "unknown",
            task=task,
            detail=detail,
            success=success,
            metadata=metadata or {},
        )
        self._events.append(event)

    def fetch(self, student_id: Optional[str] = None) -> List[LearningEvent]:
        if student_id is None:
            return list(self._events)
        return [e for e in self._events if e.student_id == student_id]
