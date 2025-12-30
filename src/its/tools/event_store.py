"""学习事件日志存储。"""
from __future__ import annotations

from collections import Counter
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

    def stats_by_task(self, student_id: Optional[str] = None) -> Dict[AgentTask, Dict[str, float]]:
        events = self.fetch(student_id)
        bucket: Dict[AgentTask, List[LearningEvent]] = {}
        for event in events:
            bucket.setdefault(event.task, []).append(event)
        stats: Dict[AgentTask, Dict[str, float]] = {}
        for task, items in bucket.items():
            total = len(items)
            success_count = sum(1 for i in items if i.success)
            stats[task] = {
                "total": float(total),
                "success_rate": (success_count / total) if total else 0.0,
                "detail_counts": dict(Counter(i.detail for i in items)),
            }
        return stats
