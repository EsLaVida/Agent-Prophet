import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.agent import app as langgraph_app

app = FastAPI(
    title="Forecasting Agent API",
    description="API для прогнозирования продаж, поездок и цен с использованием Prophet и LangGraph",
    version="1.0.0"
)

class UserMessage(BaseModel):
    text: str
    session_id: str = None 

@app.post("/chat")
async def chat_endpoint(payload: UserMessage):
    
    thread_id = payload.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=payload.text)]}
    final_state = await langgraph_app.ainvoke(inputs, config=config)
    last_message = final_state["messages"][-1]

    return {
        "reply": last_message.content,
        "session_id": thread_id,
        "payload": {
            "value": final_state.get("prediction_result"),
            "series": final_state.get("series_name"),
            "target_date": final_state.get("target_date")
        }
    }

if __name__ == "__main__":
    # Запуск сервера
    print("API сервер запускается на http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)