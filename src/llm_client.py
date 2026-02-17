from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Не найден OPENAI_API_KEY в файле .env")

llm = ChatOpenAI(
    model="xiaomi/mimo-v2-flash", 
    base_url="https://openrouter.ai/api/v1",
    temperature=0.4,
    api_key=OPENAI_API_KEY)

