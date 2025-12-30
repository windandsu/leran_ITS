"""Practice Agent 生成练习与反馈。"""
from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask, PracticeFeedback
from its.tools.event_store import EventStoreTool


class PracticeAgent(BaseAgent):
    """生成题目、评估答案并记录事件。"""

    def __init__(self, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="practice", **kwargs)
        self.event_store = event_store
        self.prompt = ChatPromptTemplate.from_template(
            """你是一名助教。先生成 2 道小测 + 1 道编程题，接着评估学生答案。\n"
            "题目主题: {topic}\n学生答案: {answer}\n"
            "请给出逐题评分(0-1)、提示、改进建议，并以 JSON 结构输出。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.PRACTICE:
            return self._default_respond("Practice Agent 未处理该任务。")

        answer = message.payload.get("answer", "")
        student_id = message.payload.get("student_id", "unknown")
        topic = message.payload.get("topic", "深度学习基础")
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x)
        feedback_raw = chain.invoke({"answer": answer, "topic": topic})
        content = feedback_raw if isinstance(feedback_raw, str) else str(feedback_raw)
        feedback = self._parse_feedback(content)
        score = feedback.score if feedback else 0.0
        self.event_store.log(
            task=AgentTask.PRACTICE,
            detail="练习批改",
            success=score >= 0.6,
            metadata={"answer": answer, "student_id": student_id, "score": score, "topic": topic},
        )
        return AgentResponse(content=content, updates={"score": score, "hints": feedback.hints if feedback else []})

    def _parse_feedback(self, content: str) -> PracticeFeedback | None:
        """将 LLM 输出解析为结构化反馈，便于分析。"""

        try:
            data = json.loads(content)
            return PracticeFeedback(
                score=float(data.get("score", 0.0)),
                hints=data.get("hints", []),
                strengths=data.get("strengths", []),
                improvements=data.get("improvements", []),
            )
        except Exception:
            return None
