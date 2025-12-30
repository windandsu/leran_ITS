"""Analytics Agent 汇总学生数据。"""
from __future__ import annotations

from statistics import mean
from typing import Dict, List

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask, LearningEvent
from its.tools.event_store import EventStoreTool


class AnalyticsAgent(BaseAgent):
    """提供个体与群体分析报表。"""

    def __init__(self, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="analytics", **kwargs)
        self.event_store = event_store

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.ANALYTICS:
            return self._default_respond("Analytics Agent 未处理该任务。")

        student_id = message.payload.get("student_id")
        events = self.event_store.fetch(student_id=student_id)
        score = self._compute_success_rate(events)
        content = f"学生 {student_id} 成功率 {score:.2%}, 事件总数 {len(events)}"
        return AgentResponse(content=content, updates={"events": events, "success_rate": score})

    def _compute_success_rate(self, events: List[LearningEvent]) -> float:
        if not events:
            return 0.0
        return mean(1.0 if e.success else 0.0 for e in events)
