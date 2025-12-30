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
        self.version = 1
        self.pending_changes: List[Document] = []
        self.prompt = ChatPromptTemplate.from_template(
            """审查以下文档摘要，标记重复、错误并给出元数据建议: {summaries}"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.CURATE:
            return self._default_respond("Curator Agent 未处理该任务。")

        docs: List[Document] = message.payload.get("documents", [])
        summaries = [doc.page_content[:160] for doc in docs]
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["summaries"])
        review = chain.invoke({"summaries": summaries})
        content = review if isinstance(review, str) else str(review)
        curated_docs = []
        seen_signatures = {d.metadata.get("sha") for d in self.pending_changes if d.metadata.get("sha")}
        for doc in docs:
            signature = doc.metadata.get("sha") or doc.page_content[:64]
            if signature in seen_signatures:
                continue
            doc.metadata.update({"curated": True, "version": self.version})
            curated_docs.append(doc)
            seen_signatures.add(signature)
        self.pending_changes.extend(curated_docs)
        self.version += 1
        return AgentResponse(
            content=content,
            updates={"curated_docs": curated_docs, "pending_versions": self.pending_changes, "version": self.version},
        )
