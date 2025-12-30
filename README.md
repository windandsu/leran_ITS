# 多 Agent 个性化深度学习系统

本仓库提供一个基于 **LangChain**、**LangGraph** 与 **GraphRAG** 的多 Agent 框架，面向深度学习教育场景，覆盖学生与教师双侧需求。

## 架构概览
- **学生端 Agent**
  - TutorAgent：解释概念、示例与引用，基于历史提问做上下文反馈。【`src/its/agents/student/tutor.py`】
  - PracticeAgent：生成练习、批改并返回结构化评分。【`src/its/agents/student/practice.py`】
  - PlannerAgent：根据画像与练习表现生成动态计划（含间隔复习）。【`src/its/agents/student/planner.py`】
  - SummarizerAgent：聚合事件生成周报并更新画像。【`src/its/agents/student/summarizer.py`】
- **教师端 Agent**
  - IngestionAgent：导入 PDF/Markdown/HTML 等资源并切分。【`src/its/agents/teacher/ingestion.py`】
  - KGBuilderAgent：利用 GraphRAG 构图并支持命名空间更新。【`src/its/agents/teacher/kg_builder.py`】
  - CuratorAgent：去重、版本控制、元数据完善。【`src/its/agents/teacher/curator.py`】
  - AnalyticsAgent：事件聚合、任务成功率可视化。【`src/its/agents/teacher/analytics.py`】
- **协调器**
  - Orchestrator：基于 LangGraph/可扩展的路由中心。【`src/its/orchestrator.py`】
- **共享工具**
  - RetrieverTool：文本+知识图谱检索。【`src/its/tools/retriever.py`】
  - ProfileStoreTool：学生画像存储。【`src/its/tools/profile_store.py`】
  - EventStoreTool：学习事件日志。【`src/its/tools/event_store.py`】
  - CitationTool：统一引用格式。【`src/its/tools/citation.py`】
- **数据模型**
  - 统一数据结构与任务类型定义。【`src/its/data_models.py`】

## 快速上手
1. 初始化工具与 Agent：
```python
from its.tools.retriever import RetrieverTool
from its.tools.profile_store import ProfileStoreTool
from its.tools.event_store import EventStoreTool
from its.tools.citation import CitationTool
from its.agents.student.tutor import TutorAgent
from its.orchestrator import Orchestrator

retriever = RetrieverTool()
profile_store = ProfileStoreTool()
event_store = EventStoreTool()
citation_tool = CitationTool()
tutor = TutorAgent(retriever, citation_tool, profile_store, event_store)
practice = PracticeAgent(event_store)
planner = PlannerAgent(profile_store, event_store)
summarizer = SummarizerAgent(profile_store, event_store)
# 教师端 agent 省略示例
orchestrator = Orchestrator(tutor, practice, planner, summarizer, ingestion, kg_builder, curator, analytics)
```
2. 通过 `Orchestrator.dispatch` 路由任务：
```python
from its.data_models import AgentMessage, AgentTask

message = AgentMessage(sender="ui", recipient="tutor", task=AgentTask.EXPLAIN, payload={"question": "什么是Transformer?"})
response = orchestrator.dispatch(message)
print(response.content, response.citations)
```

> 代码目前聚焦于模块接口设计，实际部署时可替换内存实现为数据库、消息队列或微服务。若已使用微软 GraphRAG 完成知识图谱构建，可直接将 `GraphRAG` 实例传入 `RetrieverTool` 以复用现有索引和检索链路。
