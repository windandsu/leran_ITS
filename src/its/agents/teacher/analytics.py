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
        stats = self.event_store.stats_by_task(student_id=student_id)
        score = self._compute_success_rate(events)
        chart = self._render_bars(stats)
        content = (
            f"学生 {student_id or '全部'} 成功率 {score:.2%}, 事件总数 {len(events)}\n"
            f"按任务统计:\n{chart}"
        )
        return AgentResponse(content=content, updates={"events": events, "success_rate": score, "task_stats": stats})

    def _compute_success_rate(self, events: List[LearningEvent]) -> float:
        if not events:
            return 0.0
        return mean(1.0 if e.success else 0.0 for e in events)

    def _render_bars(self, stats: Dict[AgentTask, Dict[str, float]]) -> str:
        lines = []
        for task, detail in stats.items():
            rate = detail.get("success_rate", 0.0)
            bar = "█" * int(rate * 10)
            lines.append(f"- {task.value}: {bar or '▫️'} {rate:.0%} ({int(detail.get('total', 0))})")
        return "\n".join(lines)
