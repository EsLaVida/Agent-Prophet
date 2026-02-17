# Агент для прогнозирования временных рядов

Проект делает прогнозы для трех рядов (`sales`, `trips`, `price`) через диалог с LLM и расчет в Prophet.

## Что умеет

- Принимает запросы на естественном языке (русский).
- Определяет нужный ряд и дату прогноза через tool-call.
- Строит прогноз через `Prophet`.
- Работает в двух режимах: CLI и API.

## Технологии

- `langgraph`, `langchain-core`, `langchain-openai`
- `prophet`, `pandas`, `numpy`
- `fastapi`, `uvicorn`

## Требования

- Python 3.11+
- Ключ OpenRouter в `.env`:

```env
OPENAI_API_KEY=your_openrouter_key
```

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Данные

Агент читает CSV из папки `tests/`:

- `tests/sales.csv`
- `tests/trips.csv`
- `tests/price.csv`

Формат каждого файла:

```csv
ds,y
2024-01-01,100
2024-01-02,120
```

- `ds` — дата
- `y` — значение ряда

## Запуск

### 1) CLI (основная точка входа)

В корне проекта должен быть только `main.py` как Python entrypoint.

```bash
python main.py
```

Примеры запросов:

- `Сколько будет продаж 2026-02-20?`
- `Прогноз trips на послезавтра`
- `Какая будет price через неделю?`

### 2) API (локальные тесты)

Файл запуска API находится в `tests/`.

```bash
python tests/run_api.py
```

Сервер: `http://127.0.0.1:8000`

Пример запроса:

```bash
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Сколько будет продаж 2026-02-20?\"}"
```

Пример ответа:

```json
{
  "reply": "Согласно моим расчетам, sales на дату 2026-02-20 составит примерно 172.",
  "session_id": "uuid",
  "payload": {
    "value": 172.3,
    "series": "sales",
    "target_date": "2026-02-20"
  }
}
```

## Структура проекта

```text
.
├── main.py
├── requirements.txt
├── config/
│   ├── prompts.py
│   └── settings.py
├── src/
│   ├── agent.py
│   ├── llm_client.py
│   └── tools.py
└── tests/
    ├── run_api.py
    ├── creaty.py
    ├── sales.csv
    ├── trips.csv
    └── price.csv
```

