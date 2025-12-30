"""KG Builder Agent 使用 GraphRAG 构建知识图谱。"""
from __future__ import annotations

from typing import List

from graphrag.index import GraphRAG
from langchain_core.documents import Document

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask


class KGBuilderAgent(BaseAgent):
    """从文档构建与更新知识图谱。"""

    def __init__(self, graphrag: GraphRAG, **kwargs) -> None:
        super().__init__(name="kg_builder", **kwargs)
        self.graphrag = graphrag

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.BUILD_KG:
            return self._default_respond("KG Builder Agent 未处理该任务。")

        docs: List[Document] = message.payload.get("documents", [])
        self.graphrag.add_documents(docs)
        summary = self.graphrag.describe_graph()
        return AgentResponse(content="知识图谱已更新", updates={"graph_summary": summary})
