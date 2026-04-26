"""
Configuration settings for the Ariadne AI assistant.

This file defines the application settings using Pydantic's BaseSettings,
loading values from environment variables or providing defaults.

Author: Georgios Giannopoulos
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Application-wide settings for Ariadne.

    Attributes:
        google_api_key (str): API key for Google Gemini services.
        llm_model (str): The main Gemini model identifier.
        lite_llm_model (str): The lightweight Gemini model identifier.
        embed_model (str): The Gemini embedding model identifier.
        hybrid_search_alpha (float): Alpha parameter for hybrid search (0 to 1).
        semantic_threshold (float): Similarity threshold for retrieved nodes.
        semantic_cache_threshold (float): Similarity threshold for semantic cache hits.
        similarity_top_k (int): Number of top results to retrieve from vector store.
        nodes_sent_to_llm (int): Number of retrieved nodes to include in LLM context.
        redis_url (str): Connection URL for the Redis server.
        qdrant_host (str): Hostname for the Qdrant server.
        qdrant_port (int): Port number for the Qdrant server.
        qdrant_collection (str): Name of the Qdrant collection.
        cache_ttl_seconds (int): Time-to-live for cache entries in seconds.
        model_config (SettingsConfigDict): Pydantic configuration for settings.
    """
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
    model_config: SettingsConfigDict = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# singleton
settings: AppSettings = AppSettings()
