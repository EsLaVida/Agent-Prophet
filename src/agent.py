from typing import Annotated, Optional, Sequence, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from prophet import Prophet

from config.prompts import sys_msg
from src.llm_client import LLMClient
from src.tools import get_prediction


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    prediction_result: Optional[float]
    target_date: Optional[str]
    series_name: Optional[str]


class ForecastingAgent:
    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.llm = self.llm_client.get_client()
        self.tools_list = [get_prediction]
        self.tool_node = ToolNode(tools=self.tools_list)
        self.graph = self._build_graph()
        self.memory_checkpointer = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory_checkpointer)

    def _assistant(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        normalized_messages = []
        for msg in messages:
            if normalized_messages and normalized_messages[-1].type == msg.type == "human":
                normalized_messages[-1] = msg
            else:
                normalized_messages.append(msg)

        llm_with_tools = self.llm.bind_tools(self.tools_list)
        ai_msg = llm_with_tools.invoke([sys_msg] + normalized_messages)
        return {"messages": [ai_msg]}

    def _predictor_node(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]

        tool_call = next(
            (
                tc
                for tc in getattr(last_message, "tool_calls", [])
                if tc["name"] == "get_prediction"
            ),
            None,
        )
        if not tool_call:
            return state

        args = tool_call["args"]
        series_name = args.get("series_name")
        target_date = args.get("target_date")

        df = pd.read_csv(f"tests/{series_name}.csv")
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        call_id = tool_call["id"]

        print("[LOGS: Prophet] 🧠 Обучаю математическую модель...")
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(df)
        future = pd.DataFrame({"ds": [pd.to_datetime(target_date)]})
        forecast = model.predict(future)
        raw_val = float(forecast.iloc[0]["yhat"])
        print("[LOGS: Prophet] ✅ Расчет окончен. Передаю данные агенту...")

        tool_content = (
            f"Согласно моим расчетам, {series_name} на дату {target_date} составит примерно {raw_val}."
        )
        return {
            "prediction_result": raw_val,
            "target_date": target_date,
            "series_name": series_name,
            "messages": [ToolMessage(tool_call_id=call_id, content=tool_content)],
        }

    def _route(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return END

        for call in last.tool_calls:
            if call["name"] == "get_prediction":
                return "predictor"
        return "tools"

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._assistant)
        graph.add_node("predictor", self._predictor_node)
        graph.add_node("tools", self.tool_node)
        graph.add_conditional_edges(
            "agent",
            self._route,
            {"predictor": "predictor", "tools": "tools", END: END},
        )
        graph.add_edge(START, "agent")
        graph.add_edge("predictor", "agent")
        graph.add_edge("tools", "agent")
        return graph


agent = ForecastingAgent()
app = agent.app