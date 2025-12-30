"""Summarizer Agent 生成周报与画像更新。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.event_store import EventStoreTool
from its.tools.profile_store import ProfileStoreTool


class SummarizerAgent(BaseAgent):
    """汇总学习进展并更新画像。"""

    def __init__(self, profile_store: ProfileStoreTool, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="summarizer", **kwargs)
        self.profile_store = profile_store
        self.event_store = event_store
        self.prompt = ChatPromptTemplate.from_template(
            """请用中文输出周报，包含三部分: \n1) 学生亮点 2) 需要加强的概念 3) 下周行动建议。\n"
            "输入数据:\n- 最近事件: {events}\n- 画像笔记: {notes}\n- 历史问题: {history}\n"
            "格式: 项目符号列出要点，并给出 1-2 个可执行任务。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.SUMMARIZE:
            return self._default_respond("Summarizer Agent 未处理该任务。")

        student_id = message.payload["student_id"]
        events = self.event_store.fetch(student_id=student_id)
        profile = self.profile_store.get(student_id)
        notes = profile.progress_notes[-5:]
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x)
        report = chain.invoke({"events": events, "notes": notes, "history": profile.recent_questions})
        self.profile_store.append_history(student_id, "weekly_summary")
        content = report if isinstance(report, str) else str(report)
        return AgentResponse(content=content, updates={"last_summary": content, "events": events})
