from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage


@tool
def get_prediction(series_name: str, target_date: str):
    """
    Вызывает модель прогнозирования для указанного ряда и даты.
    series_name: ['sales', 'trips', 'price']
    target_date: дата в формате YYYY-MM-DD
    """
    # Мы можем вернуть просто текст, так как основные расчеты сделает predictor_node, перехватив вызов.
    return f"Подготовка прогноза для {series_name}..."



tools_list = [get_prediction]

tool_node = ToolNode(tools=tools_list)