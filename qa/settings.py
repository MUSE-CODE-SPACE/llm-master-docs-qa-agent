"""Configuration loaded from environment / .env. / 환경변수·.env에서 설정 로드."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    chroma_dir: str = "./chroma_db"
    embedding_model: str = "text-embedding-3-small"
    provider: str = "openai"  # openai | anthropic | ollama
    llm_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434/v1"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    collection_name: str = "company_docs"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
