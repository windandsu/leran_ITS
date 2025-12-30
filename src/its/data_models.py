"""核心数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentTask(str, Enum):
    """Agent 可执行的任务类型。"""

    EXPLAIN = "explain"
    PRACTICE = "practice"
    PLAN = "plan"
    SUMMARIZE = "summarize"
    INGEST = "ingest"
    BUILD_KG = "build_kg"
    CURATE = "curate"
    ANALYTICS = "analytics"


@dataclass
class StudentProfile:
    """学生学习画像。"""

    student_id: str
    goals: List[str]
    skills: Dict[str, float]
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    recent_questions: List[str] = field(default_factory=list)
    schedule: List["PlanItem"] = field(default_factory=list)
    progress_notes: List[str] = field(default_factory=list)


@dataclass
class PlanItem:
    """学习计划条目，支持间隔重复。"""

    title: str
    due: datetime
    interval_days: int
    resources: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    review: bool = False


@dataclass
class PracticeFeedback:
    """练习评估反馈。"""

    score: float
    hints: List[str]
    strengths: List[str]
    improvements: List[str]


@dataclass
class LearningEvent:
    """学习事件记录。"""

    student_id: str
    task: AgentTask
    detail: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    """检索请求。"""

    question: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """检索结果。"""

    passages: List[str]
    citations: List[str]


@dataclass
class AgentMessage:
    """Agent 间通信消息。"""

    sender: str
    recipient: str
    task: AgentTask
    payload: Dict[str, Any]


@dataclass
class AgentResponse:
    """Agent 输出结构。"""

    content: str
    citations: List[str] = field(default_factory=list)
    updates: Dict[str, Any] = field(default_factory=dict)

