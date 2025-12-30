"""Agent 基类与公共逻辑。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models import BaseChatModel

from its.data_models import AgentMessage, AgentResponse


class BaseAgent(ABC):
    """统一的 Agent 抽象。"""

    def __init__(self, name: str, llm: Optional[BaseChatModel] = None) -> None:
        self.name = name
        self.llm = llm

    @abstractmethod
    def handle(self, message: AgentMessage) -> AgentResponse:
        """处理来自协调器的消息。"""

    def _default_respond(self, content: str) -> AgentResponse:
        return AgentResponse(content=content)
