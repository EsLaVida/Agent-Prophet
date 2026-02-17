import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import app as langgraph_app


def run_cli():
    print("\n" + "="*50)
    print("📈 FORECASTING AGENT (PROPHET) ПРИВЕТСТВУЕТ ВАС")
    print("="*50)
    print("(Доступные ряды: sales, trips, price)")
    print("(Введите 'стоп' для выхода)")

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
            final_state = langgraph_app.invoke(inputs, config=config)
            last_message = final_state["messages"][-1]
            
            if isinstance(last_message, AIMessage) and last_message.content:
                print(f"\n🤖 Ассистент: {last_message.content}")
            else:
                print("\n🤖 Ассистент: Вычисляю параметры прогноза...")
            
        except Exception as e:
            print(f"!!! Произошла ошибка: {e}")
            print("Проверьте наличие CSV-файлов или формат даты.")

if __name__ == "__main__":
    run_cli()

