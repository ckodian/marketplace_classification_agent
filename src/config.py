from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    database_url: str = "postgresql+asyncpg://user:password@localhost:5533/pgvector"
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_embedder: str = "nomic-embed-text"
    ollama_embedding_dimension: int = 768
    ollama_text_model: str = "qwen2.5:7b"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_max_batch_embedding: int = 10

    timeout_llm_request_seconds: int = 300
    # Ollama runs locally — first-token latency for a freshly loaded
    # model can be high (model load + warmup), so we allow 5 minutes.
    timeout_llm_ollama_seconds: int = 300
    # Generic external HTTP — embeddings, sample fetches, news API.
    timeout_external_http_seconds: int = 30

    cors_origins: list[str] = ["*"]
    env: str = "development"


settings = Settings()