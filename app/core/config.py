from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # ChromaDB connection settings
    CHROMA_HOST: str
    CHROMA_PORT: int
    VECTOR_STORE_PATH: str

    # Retrieval configuration
    RETRIEVAL_TOP_K: int = 3

    # Ollama model configuration
    EMBEDDING_MODEL_NAME: str
    OLLAMA_MODEL: str
    OLLAMA_BASE_URL: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
