import uuid
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import app as langgraph_app


# --- КОНСОЛЬНЫЙ РЕЖИМ (CLI) ---
def run_cli():
    print("\n" + "="*50)
    print("📈 FORECASTING AGENT (PROPHET) ПРИВЕТСТВУЕТ ВАС")
    print("="*50)
    print("(Доступные ряды: sales, trips, price)")
    print("(Введите 'стоп' для выхода)")

    # Создаем ID сессии для хранения контекста в рамках одного запуска
    session_id = str(uuid.uuid4())
    print(f"🆔 ID твоей сессии: {session_id}")

    while True:
        user_input = input("\n👤 Вы: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["стоп", "exit", "quit", "выход"]:
            print("👋 До свидания! Аналитика завершена.")
            break

        try:
            config = {
                "configurable": {"thread_id": session_id}
            }

            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Прогон через граф
            final_state = langgraph_app.invoke(inputs, config=config)

            # Получаем последнее сообщение
            last_message = final_state["messages"][-1]
            
            if isinstance(last_message, AIMessage) and last_message.content:
                print(f"\n🤖 Ассистент: {last_message.content}")
            else:
                # Если вдруг последнее сообщение — это вызов инструмента (хотя граф должен вернуть AIMessage)
                print("\n🤖 Ассистент: Вычисляю параметры прогноза...")
            
        except Exception as e:
            print(f"!!! Произошла ошибка: {e}")
            print("Проверьте наличие CSV-файлов или формат даты.")

if __name__ == "__main__":
    #  API : uvicorn main:app --reload
    run_cli()

