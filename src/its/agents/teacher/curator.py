"""Curator/QA Agent 负责质量控制与去重。"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask


class CuratorAgent(BaseAgent):
    """检查准确性、重复与元数据。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(name="curator", **kwargs)
        self.prompt = ChatPromptTemplate.from_template(
            """审查以下文档摘要，标记重复、错误并给出元数据建议: {summaries}"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.CURATE:
            return self._default_respond("Curator Agent 未处理该任务。")

        docs: List[Document] = message.payload.get("documents", [])
        summaries = [doc.page_content[:120] for doc in docs]
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["summaries"])
        review = chain.invoke({"summaries": summaries})
        content = review if isinstance(review, str) else str(review)
        enriched = [doc.metadata.update({"curated": True}) or doc for doc in docs]
        return AgentResponse(content=content, updates={"curated_docs": enriched})
