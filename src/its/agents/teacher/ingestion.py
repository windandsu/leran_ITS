"""Ingestion Agent 导入外部教育资源。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask


class IngestionAgent(BaseAgent):
    """加载文本与论文等资源，输出结构化 Document 列表。"""

    def __init__(self, target_dir: Path, **kwargs) -> None:
        super().__init__(name="ingestion", **kwargs)
        self.target_dir = target_dir

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.INGEST:
            return self._default_respond("Ingestion Agent 未处理该任务。")

        sources: List[str] = message.payload.get("sources", [])
        documents: List[Document] = []
        for src in sources:
            path = Path(src)
            loader = TextLoader(str(path))
            documents.extend(loader.load())
        content = f"已导入 {len(documents)} 条文档"
        return AgentResponse(content=content, updates={"documents": documents})
