import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    llm_model: str = "xiaomi/mimo-v2-flash"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.4

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Не найден OPENAI_API_KEY в файле .env")

        model = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-flash")
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))

        return cls(
            openai_api_key=api_key,
            llm_model=model,
            llm_base_url=base_url,
            llm_temperature=temperature,
        )


settings = Settings.from_env()
