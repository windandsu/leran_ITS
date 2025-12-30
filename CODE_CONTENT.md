# 代码内容汇总

以下为仓库核心模块的当前源码，直接展开便于查阅。

## `src/its/data_models.py`
```python
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


```

## `src/its/agents/base.py`
```python
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

```

## `src/its/agents/student/tutor.py`
```python
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

```

## `src/its/agents/student/practice.py`
```python
"""Practice Agent 生成练习与反馈。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.event_store import EventStoreTool


class PracticeAgent(BaseAgent):
    """生成题目、评估答案并记录事件。"""

    def __init__(self, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="practice", **kwargs)
        self.event_store = event_store
        self.prompt = ChatPromptTemplate.from_template(
            """根据学生输入生成 3 道深度学习练习，包含选择题与编程题。学生答案: {answer}\n给出评分与改进建议。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.PRACTICE:
            return self._default_respond("Practice Agent 未处理该任务。")

        answer = message.payload.get("answer", "")
        student_id = message.payload.get("student_id", "unknown")
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["answer"])
        feedback = chain.invoke({"answer": answer})
        self.event_store.log(
            task=AgentTask.PRACTICE,
            detail="练习批改",
            success=True,
            metadata={"answer": answer, "student_id": student_id},
        )
        content = feedback if isinstance(feedback, str) else str(feedback)
        return AgentResponse(content=content)

```

## `src/its/agents/student/planner.py`
```python
"""Planner Agent 负责个性化学习计划。"""
from __future__ import annotations

from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask, StudentProfile
from its.tools.profile_store import ProfileStoreTool


class PlannerAgent(BaseAgent):
    """根据目标与历史生成计划。"""

    def __init__(self, profile_store: ProfileStoreTool, **kwargs) -> None:
        super().__init__(name="planner", **kwargs)
        self.profile_store = profile_store
        self.prompt = ChatPromptTemplate.from_template(
            """学生画像: {profile}\n请生成 7 天深度学习学习日程，包含每日主题与资源。"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.PLAN:
            return self._default_respond("Planner Agent 未处理该任务。")

        student_id = message.payload["student_id"]
        profile = self.profile_store.get(student_id)
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["profile"])
        plan = chain.invoke({"profile": profile})
        schedule = self._build_schedule()
        self.profile_store.update(student_id, {"schedule": schedule})
        content = plan if isinstance(plan, str) else str(plan)
        return AgentResponse(content=content, updates={"schedule": schedule})

    def _build_schedule(self) -> dict:
        base = datetime.utcnow()
        return {f"day_{i}": (base + timedelta(days=i)).isoformat() for i in range(7)}

```

## `src/its/agents/student/summarizer.py`
```python
"""Summarizer Agent 生成周报与画像更新。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask
from its.tools.profile_store import ProfileStoreTool


class SummarizerAgent(BaseAgent):
    """汇总学习进展并更新画像。"""

    def __init__(self, profile_store: ProfileStoreTool, **kwargs) -> None:
        super().__init__(name="summarizer", **kwargs)
        self.profile_store = profile_store
        self.prompt = ChatPromptTemplate.from_template(
            """基于以下活动总结学生一周进展并提出建议: {events}"""
        )

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.SUMMARIZE:
            return self._default_respond("Summarizer Agent 未处理该任务。")

        student_id = message.payload["student_id"]
        events = message.payload.get("events", [])
        chain = self.prompt | self.llm if self.llm else RunnableLambda(lambda x: x["events"])
        report = chain.invoke({"events": events})
        self.profile_store.append_history(student_id, "weekly_summary")
        content = report if isinstance(report, str) else str(report)
        return AgentResponse(content=content, updates={"last_summary": content})

```

## `src/its/agents/teacher/ingestion.py`
```python
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

```

## `src/its/agents/teacher/kg_builder.py`
```python
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

```

## `src/its/agents/teacher/curator.py`
```python
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

```

## `src/its/agents/teacher/analytics.py`
```python
"""Analytics Agent 汇总学生数据。"""
from __future__ import annotations

from statistics import mean
from typing import Dict, List

from its.agents.base import BaseAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask, LearningEvent
from its.tools.event_store import EventStoreTool


class AnalyticsAgent(BaseAgent):
    """提供个体与群体分析报表。"""

    def __init__(self, event_store: EventStoreTool, **kwargs) -> None:
        super().__init__(name="analytics", **kwargs)
        self.event_store = event_store

    def handle(self, message: AgentMessage) -> AgentResponse:
        if message.task != AgentTask.ANALYTICS:
            return self._default_respond("Analytics Agent 未处理该任务。")

        student_id = message.payload.get("student_id")
        events = self.event_store.fetch(student_id=student_id)
        score = self._compute_success_rate(events)
        content = f"学生 {student_id} 成功率 {score:.2%}, 事件总数 {len(events)}"
        return AgentResponse(content=content, updates={"events": events, "success_rate": score})

    def _compute_success_rate(self, events: List[LearningEvent]) -> float:
        if not events:
            return 0.0
        return mean(1.0 if e.success else 0.0 for e in events)

```

## `src/its/orchestrator.py`
```python
"""Orchestrator/Coordinator 负责路由任务。"""
from __future__ import annotations

from typing import Dict

from langgraph.graph import END, StateGraph

from its.agents.student.planner import PlannerAgent
from its.agents.student.practice import PracticeAgent
from its.agents.student.summarizer import SummarizerAgent
from its.agents.student.tutor import TutorAgent
from its.agents.teacher.analytics import AnalyticsAgent
from its.agents.teacher.curator import CuratorAgent
from its.agents.teacher.ingestion import IngestionAgent
from its.agents.teacher.kg_builder import KGBuilderAgent
from its.data_models import AgentMessage, AgentResponse, AgentTask


class Orchestrator:
    """通过 LangGraph 构建的协调器，确保 Agent 协同。"""

    def __init__(
        self,
        tutor: TutorAgent,
        practice: PracticeAgent,
        planner: PlannerAgent,
        summarizer: SummarizerAgent,
        ingestion: IngestionAgent,
        kg_builder: KGBuilderAgent,
        curator: CuratorAgent,
        analytics: AnalyticsAgent,
    ) -> None:
        self.agents: Dict[AgentTask, object] = {
            AgentTask.EXPLAIN: tutor,
            AgentTask.PRACTICE: practice,
            AgentTask.PLAN: planner,
            AgentTask.SUMMARIZE: summarizer,
            AgentTask.INGEST: ingestion,
            AgentTask.BUILD_KG: kg_builder,
            AgentTask.CURATE: curator,
            AgentTask.ANALYTICS: analytics,
        }
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建基于 LangGraph 的简单路由图。"""

        # Graph 状态仅保留消息与 agent 响应，确保节点输出可追踪
        graph: StateGraph = StateGraph(dict)

        def route(state: Dict) -> str:
            message: AgentMessage = state["message"]
            return message.task.value

        graph.add_node("router", lambda state: state)
        for task, agent in self.agents.items():
            graph.add_node(
                task.value, lambda state, agent=agent: {"message": state["message"], "response": agent.handle(state["message"])}
            )
        graph.add_conditional_edges(
            "router",
            route,
            {task.value: task.value for task in self.agents.keys()},
            [],
        )
        for task in self.agents.keys():
            graph.add_edge(task.value, END)
        graph.set_entry_point("router")
        return graph.compile()

    def dispatch(self, message: AgentMessage) -> AgentResponse:
        if message.task not in self.agents:
            return AgentResponse(content="未找到匹配 Agent")

        result = self.graph.invoke({"message": message})
        return result.get("response", AgentResponse(content="无响应"))

```

## `src/its/tools/retriever.py`
```python
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

```

## `src/its/tools/profile_store.py`
```python
"""管理学生画像的轻量级存储。"""
from __future__ import annotations

from typing import Dict

from its.data_models import StudentProfile


class ProfileStoreTool:
    """简单的内存画像存储，未来可替换为数据库。"""

    def __init__(self) -> None:
        self._profiles: Dict[str, StudentProfile] = {}

    def get(self, student_id: str) -> StudentProfile:
        return self._profiles.setdefault(
            student_id, StudentProfile(student_id=student_id, goals=[], skills={}, preferences={})
        )

    def update(self, student_id: str, updates: dict) -> StudentProfile:
        profile = self.get(student_id)
        for key, value in updates.items():
            setattr(profile, key, value)
        return profile

    def append_history(self, student_id: str, record: str) -> None:
        profile = self.get(student_id)
        profile.history.append(record)

```

## `src/its/tools/event_store.py`
```python
"""学习事件日志存储。"""
from __future__ import annotations

from typing import Dict, List, Optional

from its.data_models import AgentTask, LearningEvent


class EventStoreTool:
    """记录并查询学习事件，可替换为消息队列或数据库。"""

    def __init__(self) -> None:
        self._events: List[LearningEvent] = []

    def log(self, task: AgentTask, detail: str, success: bool = True, metadata: Optional[Dict] = None) -> None:
        event = LearningEvent(
            student_id=metadata.get("student_id", "unknown") if metadata else "unknown",
            task=task,
            detail=detail,
            success=success,
            metadata=metadata or {},
        )
        self._events.append(event)

    def fetch(self, student_id: Optional[str] = None) -> List[LearningEvent]:
        if student_id is None:
            return list(self._events)
        return [e for e in self._events if e.student_id == student_id]

```

## `src/its/tools/citation.py`
```python
"""引用工具，确保输出可追溯。"""
from __future__ import annotations

from typing import List


class CitationTool:
    """将来源列表转换为统一的引用格式。"""

    def attach(self, sources: List[str]) -> List[str]:
        return [f"来源:{src}" for src in sources]

```

