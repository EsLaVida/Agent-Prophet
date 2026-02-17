from langchain_openai import ChatOpenAI

from config.settings import Settings, settings


class LLMClient:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.client = ChatOpenAI(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            temperature=self.config.llm_temperature,
            api_key=self.config.openai_api_key,
        )

    def get_client(self) -> ChatOpenAI:
        return self.client

