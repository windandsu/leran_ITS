"""统一检索工具，结合文本与知识图谱。"""
from __future__ import annotations

from typing import List, Optional

from graphrag.index import GraphRAG
from langchain_core.vectorstores import VectorStoreRetriever

from its.data_models import RetrievalQuery, RetrievalResult


class RetrieverTool:
    """提供多模态检索 (文本 + 图谱)。"""

    def __init__(
        self, text_retriever: Optional[VectorStoreRetriever] = None, graph: Optional[GraphRAG] = None
    ) -> None:
        self.text_retriever = text_retriever
        self.graph = graph

    def search(self, question: str) -> RetrievalResult:
        query = RetrievalQuery(question=question)
        passages: List[str] = []
        citations: List[str] = []
        if self.text_retriever:
            docs = self.text_retriever.get_relevant_documents(query.question)
            passages.extend([d.page_content for d in docs])
            citations.extend([d.metadata.get("source", "text") for d in docs])
        # 直接连接已构建完成的 GraphRAG 索引，适配常见查询接口
        if self.graph:
            graph_results = self._graph_query(query.question)
            passages.extend([res["text"] for res in graph_results])
            citations.extend([res.get("source", "graph") for res in graph_results])
        return RetrievalResult(passages=passages, citations=citations)

    def _graph_query(self, question: str) -> List[dict]:
        """兼容 GraphRAG 的多种查询入口，直接复用已完成的图索引。"""

        if hasattr(self.graph, "search"):
            return getattr(self.graph, "search")(question)
        if hasattr(self.graph, "query"):
            return getattr(self.graph, "query")(question)
        if hasattr(self.graph, "generate_answer"):
            # GraphRAG v0.2+ 推荐接口，返回包含上下文的回答
            answer = getattr(self.graph, "generate_answer")(question)
            contexts = answer.get("contexts", []) if isinstance(answer, dict) else []
            return contexts
        return []
