from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loads and validates environment variables."""

    CHROMA_HOST: str
    CHROMA_PORT: int
    VECTOR_STORE_PATH: str

    RETRIEVAL_TOP_K: int = 3
    RETRIEVAL_SCORE_THRESHOLD: float

    EMBEDDING_MODEL_NAME: str
    OLLAMA_MODEL: str = "gemma3n:e2b"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
