from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    # Api keys
    google_api_key: str

    # Models
    llm_model: str = "gemini-2.5-flash" #  gemini-3.1-flash-lite-preview
    lite_llm_model: str = "gemini-2.5-flash-lite"
    embed_model: str = "gemini-embedding-2"

    # RAG parameters
    hybrid_search_alpha: float = 0.7
    semantic_threshold: float = 0.25
    semantic_cache_threshold: float = 0.92
    similarity_top_k: int = 20
    nodes_sent_to_llm: int = 15

    # Database URLs
    redis_url: str = "redis://127.0.0.1:6379"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_collection: str = "gemini-emb-rag"

    # Redis TTL
    cache_ttl_seconds: int = 86400 # 24 hours
    
    # Pydantic config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# singleton
settings = AppSettings()