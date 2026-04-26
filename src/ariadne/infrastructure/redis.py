import logging
from typing import List

import redis
from redis.asyncio import Redis as RedisAsync # for type hint in self._aclient
import redis.asyncio as redis_async
from redisvl.schema import IndexSchema
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.schema import BaseNode

from ariadne.core.config import settings

logger = logging.getLogger("RAG_Workflow")

def get_redis_clients():
    """Returns synchronous and asynchronous Redis clients."""
    client = redis.Redis.from_url(settings.redis_url)
    aclient = redis_async.Redis.from_url(settings.redis_url)
    return client, aclient

class TTLRedisVectorStore(RedisVectorStore):
    """
    A `RedisVectorStore` subclass that applies a TTL (Time-To-Live) to every
    document key written to the cache.

    By default, llama_index's RedisVectorStore writes keys with no expiry.
    This class overrides `async_add` to call Redis *EXPIRE* immediately after
    each write, ensuring stale cache entries are automatically evicted.

    Args:
        ttl (int): Expiry time in seconds applied to each written key.
                   Defaults to `settings.cache_ttl_seconds`.
    """

    def __init__(self, *args, ttl: int = settings.cache_ttl_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self._ttl = ttl
        
    async def async_add(self, nodes: List[BaseNode], **kwargs)-> List[str]:
        kwargs["ttl"] = self._ttl

        ids = await super().async_add(nodes, **kwargs)

        logger.info(
            f"Successfully saved {len(ids)} document(s) to Redis Cache", 
            extra={'ttl_applied': self._ttl}
        )
        return ids
    
cache_schema = IndexSchema.from_dict({
    "index": {
        "name": "rag_semantic_cache",
        "prefix": "semantic_cache_doc"
    },
    "fields":[
        {"type": "tag", "name": "id"},
        {"type": "tag", "name": "doc_id"},
        {"type": "text", "name": "text"},
        {"type": "text", "name": "answer"}, 
        {
            "type": "vector",
            "name": "vector",
            "attrs": {
                "dims": 3072, # gemini-embedding-2-preview dimension
                "algorithm": "FLAT",
                "distance_metric": "COSINE"
            }
        }
    ]
})
