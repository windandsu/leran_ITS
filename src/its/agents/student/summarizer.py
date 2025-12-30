"""Summarizer Agent 生成周报与画像更新。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.profile_store import ProfileStoreTool


class SummarizerAgent(BaseAgent):
    """汇总学习进展并更新画像。"""

    def __init__(self, profile_store: ProfileStoreTool, **kwargs) -> None:
        super().__init__(name="summarizer", **kwargs)
        self.profile_store = profile_store
        self.prompt = ChatPromptTemplate.from_template(
            """基于以下活动总结学生一周进展并提出建议: {events}"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.SUMMARIZE:
            return self._default_respond("Summarizer Agent 未处理该任务。")

        student_id = message.payload["student_id"]
        events = message.payload.get("events", [])
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["events"])
        report = chain.invoke({"events": events})
        self.profile_store.append_history(student_id, "weekly_summary")
        content = report if isinstance(report, str) else str(report)
        return AgentResponse(content=content, updates={"last_summary": content})
