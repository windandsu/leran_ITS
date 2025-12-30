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
