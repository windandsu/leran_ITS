"""Planner Agent 负责个性化学习计划。"""
from __future__ import annotations

from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask, StudentProfile
from its.tools.profile_store import ProfileStoreTool


class PlannerAgent(BaseAgent):
    """根据目标与历史生成计划。"""

    def __init__(self, profile_store: ProfileStoreTool, **kwargs) -> None:
        super().__init__(name="planner", **kwargs)
        self.profile_store = profile_store
        self.prompt = ChatPromptTemplate.from_template(
            """学生画像: {profile}\n请生成 7 天深度学习学习日程，包含每日主题与资源。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.PLAN:
            return self._default_respond("Planner Agent 未处理该任务。")

        student_id = message.payload["student_id"]
        profile = self.profile_store.get(student_id)
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["profile"])
        plan = chain.invoke({"profile": profile})
        schedule = self._build_schedule()
        self.profile_store.update(student_id, {"schedule": schedule})
        content = plan if isinstance(plan, str) else str(plan)
        return AgentResponse(content=content, updates={"schedule": schedule})

    def _build_schedule(self) -> dict:
        base = datetime.utcnow()
        return {f"day_{i}": (base + timedelta(days=i)).isoformat() for i in range(7)}
