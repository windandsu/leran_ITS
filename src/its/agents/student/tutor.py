"""面向学生的 Tutor Agent。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.citation import CitationTool
from its.tools.retriever import RetrieverTool


class TutorAgent(BaseAgent):
    """提供解释、示例和互动反馈。"""

    def __init__(self, retriever: RetrieverTool, citation_tool: CitationTool, **kwargs) -> None:
        super().__init__(name="tutor", **kwargs)
        self.retriever = retriever
        self.citation_tool = citation_tool
        self.prompt = ChatPromptTemplate.from_template(
            """你是一名深度学习导师，用简体中文回答。问题: {question}\n上下文: {context}\n给出简洁解释并附带必要引用。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.EXPLAIN:
            return self._default_respond("Tutor Agent 未处理该任务。")

        query = message.payload.get("question", "")
        retrieval = self.retriever.search(query)
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["context"])
        rendered = chain.invoke({"question": query, "context": "\n".join(retrieval.passages)})
        citations = self.citation_tool.attach(retrieval.citations)
        content = rendered if isinstance(rendered, str) else str(rendered)
        return AgentResponse(content=content, citations=citations)
