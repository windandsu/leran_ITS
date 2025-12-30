"""面向学生的 Tutor Agent。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.event_store import EventStoreTool
from its.tools.profile_store import ProfileStoreTool
from its.tools.citation import CitationTool
from its.tools.retriever import RetrieverTool


class TutorAgent(BaseAgent):
    """提供解释、示例和互动反馈。"""

    def __init__(
        self,
        retriever: RetrieverTool,
        citation_tool: CitationTool,
        profile_store: ProfileStoreTool,
        event_store: EventStoreTool,
        **kwargs,
    ) -> None:
        super().__init__(name="tutor", **kwargs)
        self.retriever = retriever
        self.citation_tool = citation_tool
        self.profile_store = profile_store
        self.event_store = event_store
        self.prompt = ChatPromptTemplate.from_template(
            """你是一名深度学习导师，用简体中文回答。\n"
            "历史问题: {history}\n当前问题: {question}\n上下文: {context}\n"
            "请给出分步骤解释、形象类比、最小代码示例，并标注引用。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.EXPLAIN:
            return self._default_respond("Tutor Agent 未处理该任务。")

        query = message.payload.get("question", "")
        student_id = message.payload.get("student_id", "unknown")
        profile = self.profile_store.get(student_id)
        retrieval = self.retriever.search(query)
        history = " | ".join(profile.recent_questions[-3:])
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["context"])
        rendered = chain.invoke({"question": query, "context": "\n".join(retrieval.passages), "history": history})
        citations = self.citation_tool.attach(retrieval.citations)
        content = rendered if isinstance(rendered, str) else str(rendered)
        self.profile_store.add_question(student_id, query)
        self.event_store.log(
            task=AgentTask.EXPLAIN,
            detail="讲解",
            success=True,
            metadata={"student_id": student_id, "question": query},
        )
        return AgentResponse(content=content, citations=citations)
