"""Ingestion Agent 导入外部教育资源。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for src in sources:
            path = Path(src)
            loader = self._select_loader(path)
            if not loader:
                continue
            docs = loader.load()
            documents.extend(splitter.split_documents(docs))
        content = f"已导入 {len(documents)} 条分片文档，支持 KG 构建"
        return AgentResponse(content=content, updates={"documents": documents})

    def _select_loader(self, path: Path):
        """根据文件后缀选择合适的 Loader。"""

        suffix = path.suffix.lower()
        if suffix in {".pdf"}:
            return PyPDFLoader(str(path))
        if suffix in {".md", ".markdown"}:
            return UnstructuredMarkdownLoader(str(path))
        if suffix in {".html", ".htm"}:
            return UnstructuredHTMLLoader(str(path))
        return TextLoader(str(path))
