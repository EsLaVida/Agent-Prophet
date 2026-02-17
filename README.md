# Агент прогнозирования временных рядов

Проект прогнозирует `sales`, `trips`, `price` через LLM + Prophet.

## Что внутри

- `ForecastingAgent` в `src/agent.py` (класс, собирает граф LangGraph).
- `LLMClient` в `src/llm_client.py` (инициализация модели).
- Инструменты только в `src/tools.py`.
- Настройки окружения в `config/settings.py`.

## Требования

- Python 3.11+
- `.env` в корне:

```env
OPENAI_API_KEY=your_openrouter_key
# необязательно:
# LLM_MODEL=xiaomi/mimo-v2-flash
# LLM_BASE_URL=https://openrouter.ai/api/v1
# LLM_TEMPERATURE=0.4
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Данные

CSV-файлы должны лежать в `tests/`:

- `tests/sales.csv`
- `tests/trips.csv`
- `tests/price.csv`

Формат:

```csv
ds,y
2024-01-01,100
2024-01-02,120
```

## Запуск

### CLI (основной вход)

```bash
python main.py
```

### API (локально для тестов)

```bash
python tests/run_api.py
```

API: `http://127.0.0.1:8000`

## Структура проекта

```text
.
├── main.py
├── requirements.txt
├── config/
│   ├── prompts.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── llm_client.py
│   └── tools.py
└── tests/
    ├── __init__.py
    ├── run_api.py
    ├── creaty_db.py
    ├── sales.csv
    ├── trips.csv
    └── price.csv
```
