from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from src.tools import tool_node, get_prediction
from src.llm_client import llm
from langgraph.graph.message import add_messages
from config.prompts import sys_msg
from prophet import Prophet
import pandas as pd
import logging
import os

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Флаг: ждем ли мы от пользователя "да/нет" 
    awaiting_confirmation: bool
    # Наше поле для хранения результата прогноза
    prediction_result: Optional[float] 
    target_date: Optional[str]
    series_name: Optional[str]

def assistant(state: AgentState) -> AgentState:
    # 1. Берем сообщения из стейта
    messages = state["messages"]
    # Гарантируем, что не будет двух HumanMessage подряд 
    normalized_messages = []
    for msg in messages:
        if normalized_messages and normalized_messages[-1].type == msg.type == 'human':
            normalized_messages[-1] = msg # Заменяем на более свежее
        else:
            normalized_messages.append(msg)

    # 3. Привязываем инструменты (только нужные сейчас)
    # ВАЖНО: передаем список инструментов напрямую
    llm_with_tools = llm.bind_tools([get_prediction])

    # 4. Вызов модели
    # Передаем системный промпт + нормализованную историю
    ai_msg = llm_with_tools.invoke([sys_msg] + normalized_messages)

    # 5. Возвращаем результат
    return {
        "messages": [ai_msg]
    }


# Отключаем логи cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

def predictor_node(state: AgentState) -> AgentState:
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    logging.getLogger('prophet').setLevel(logging.ERROR)

    last_message = state["messages"][-1]
    
    tool_call = next(
        (tc for tc in getattr(last_message, 'tool_calls', []) 
         if tc['name'] == 'get_prediction'), 
        None
    )
    if not tool_call:
        return state # Если вызов не найден, ничего не делаем
    # Извлекаем аргументы, подготовленные LLM
    args = tool_call['args']
    series_name = args.get("series_name")
    target_date = args.get("target_date")
    
    # агрузка данных (предполагаем, что файл {series_name}.csv существует)
    df = pd.read_csv(f"{series_name}.csv")
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    call_id = tool_call['id']
    print(f"[LOGS: Prophet] 🧠 Обучаю математическую модель...")
    # Обучение Prophet
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(df)
    
    # Прогноз
    future = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
    forecast = m.predict(future)
    # Отдаем максимально точное число. Пусть у агента будет вся информация.
    raw_val = float(forecast.iloc[0]['yhat'])

    print(f"[LOGS: Prophet] ✅ Расчет окончен. Передаю данные агенту...")

    # Формирование ответа
    tool_content = f"Согласно моим расчетам, {series_name} на дату {target_date} составит примерно {raw_val}."
    return {
        "prediction_result": raw_val,
        "target_date": target_date,
        "series_name": series_name,
        "messages": [
            ToolMessage(tool_call_id=call_id, content=tool_content)
        ]
    }
    
#графы
# 1. Инициализация графа
graph = StateGraph(AgentState)
# 2. Добавляем узлы
graph.add_node("agent", assistant)
graph.add_node("predictor", predictor_node)
graph.add_node("tools", tool_node)
# 4. Настраиваем логику переходов

def route(state: AgentState) -> str:
    last = state["messages"][-1]
    # Если инструментов нет — завершаем (ждем ввода пользователя в чат)
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return END
    # Если инструменты есть, проверяем какие именно
    for call in last.tool_calls:
        if call['name'] == 'get_prediction':
            return "predictor"
    return "tools"
graph.add_conditional_edges("agent", route, {
    "predictor": "predictor", 
    "tools": "tools", 
    END: END
    })
graph.add_edge(START, "agent")
graph.add_edge("predictor", "agent") 
graph.add_edge("tools", "agent")
# Используем Memory checkpointer - он стабильно работает
from langgraph.checkpoint.memory import MemorySaver
memory_checkpointer = MemorySaver()

app = graph.compile(checkpointer=memory_checkpointer)