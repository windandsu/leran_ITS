"""Practice Agent 生成练习与反馈。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.event_store import EventStoreTool


class PracticeAgent(BaseAgent):
    """生成题目、评估答案并记录事件。"""

    def __init__(self, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="practice", **kwargs)
        self.event_store = event_store
        self.prompt = ChatPromptTemplate.from_template(
            """根据学生输入生成 3 道深度学习练习，包含选择题与编程题。学生答案: {answer}\n给出评分与改进建议。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.PRACTICE:
            return self._default_respond("Practice Agent 未处理该任务。")

        answer = message.payload.get("answer", "")
        student_id = message.payload.get("student_id", "unknown")
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["answer"])
        feedback = chain.invoke({"answer": answer})
        self.event_store.log(
            task=AgentTask.PRACTICE,
            detail="练习批改",
            success=True,
            metadata={"answer": answer, "student_id": student_id},
        )
        content = feedback if isinstance(feedback, str) else str(feedback)
        return AgentResponse(content=content)
