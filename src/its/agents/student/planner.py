"""Planner Agent 负责个性化学习计划。"""
from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask, PlanItem, StudentProfile
from its.tools.event_store import EventStoreTool
from its.tools.profile_store import ProfileStoreTool


class PlannerAgent(BaseAgent):
    """根据目标与历史生成计划。"""

    def __init__(self, profile_store: ProfileStoreTool, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="planner", **kwargs)
        self.profile_store = profile_store
        self.event_store = event_store
        self.prompt = ChatPromptTemplate.from_template(
            """根据以下学生画像和近期表现，规划 7 天个性化学习日程（每天 1-2 个任务）。\n"
            "学生画像: {profile}\n表现概览: {stats}\n复习间隔(天): {interval}\n"
            "请标注任务难度、所需时间(<=90 分钟)和复习节点。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.PLAN:
            return self._default_respond("Planner Agent 未处理该任务。")

        student_id = message.payload["student_id"]
        profile = self.profile_store.get(student_id)
        stats = self.event_store.stats_by_task(student_id=student_id)
        interval = self._sm2_interval(stats.get(AgentTask.PRACTICE, {}).get("success_rate", 0.6))
        plan_items = self._build_schedule(profile, interval)
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["profile"])
        plan_text = chain.invoke({"profile": profile, "stats": stats, "interval": interval})
        self.profile_store.upsert_plan(student_id, plan_items)
        content = plan_text if isinstance(plan_text, str) else str(plan_text)
        return AgentResponse(content=content, updates={"schedule": plan_items, "interval": interval})

    def _build_schedule(self, profile: StudentProfile, interval: int) -> List[PlanItem]:
        base = datetime.utcnow()
        goals = profile.goals or ["卷积网络", "优化算法", "实践项目"]
        plan: List[PlanItem] = []
        for i, goal in enumerate(goals[:7]):
            due = base + timedelta(days=i)
            plan.append(
                PlanItem(
                    title=f"Day {i+1}: {goal}",
                    due=due,
                    interval_days=interval,
                    resources=[f"阅读 {goal} 教程", "完成 1 道编程练习"],
                    difficulty="hard" if profile.skills.get(goal, 0.0) < 0.4 else "medium",
                    review=i % interval == 0,
                )
            )
        return plan

    def _sm2_interval(self, quality: float) -> int:
        """简化 SM-2，质量越高间隔越长。"""

        easiness = max(1.3, 2.5 + (quality - 0.6))
        return max(1, ceil(easiness))
