import functools
import logging
from typing import List

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo
from llama_index.core.settings import Settings
from llama_index.core.rate_limiter import TokenBucketRateLimiter

from google.genai import types
from google.genai.local_tokenizer import LocalTokenizer

from ariadne.core.config import settings

logger = logging.getLogger("RAG_Workflow")


# --- Rate Limiter ---
rate_limiter = TokenBucketRateLimiter(
    requests_per_minute=100,
    tokens_per_minute=500000
)
# --- Models ---
# main llm for the final answer
llm = GoogleGenAI(
    model=settings.llm_model,
    api_key=settings.google_api_key,
    temperature=0.1, # default temp, as proposed by gemini api 
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # disable thinking
    ),
    rate_limiter=rate_limiter,
)
# Lightweight model for routing, condense questions
lite_llm = GoogleGenAI(
    model=settings.lite_llm_model,
    api_key=settings.google_api_key,
    temperature=0.1,
)
# same as above but without token restriction, used for fact memory block
fact_llm = GoogleGenAI(
    model=settings.lite_llm_model,
    api_key=settings.google_api_key,
    temperature=0.2,
)

# Embed model for retrieving
embed_model = GoogleGenAIEmbedding(
    model_name=settings.embed_model,
    api_key=settings.google_api_key,
    embedding_config=types.EmbedContentConfig(
        task_type='QUESTION_ANSWERING',
        output_dimensionality=3072,
    ),
)

# Tokenizer for counting tokens
local_tokenizer = LocalTokenizer(model_name=settings.llm_model)
def gemini_tokenizer(text: str) -> List[bytes] | List:
    """
    Tokenizes text using Google Gemini's local tokenizer.
    """
    if not text:
        return []
    
    try:
        tokens_result: types.ComputeTokensResult = local_tokenizer.compute_tokens(text)

        return tokens_result.tokens_info[0].tokens
    except Exception as e:
        print(f"Tokenizer Error: {e}")
        return []

Settings.llm = llm
Settings.embed_model = embed_model
Settings.tokenizer = gemini_tokenizer

vector_info = VectorStoreInfo(
    content_info="Course descriptions from University of Peloponnese. Semester the course is offered and category it belongs based on the official course guide.",
    metadata_info=[
        MetadataInfo(
            name='semester',
            type="str",
            description=('The semester number as a simple digit. ALWAYS use "1", "2", "3", "4", "5", "6", "7", or "8". Never use "7ου" or "1ο". The field is string, so only equals (==) should be used.'
            'CRITICAL RULE: USE THIS FILTER ONLY IF the user is explicitly asking to find or list courses taught in a specific semester.\n'
            'DO NOT USE THIS FILTER if the user is asking about general rules, graduation requirements, part-time studies (μερική φοίτηση), '
            'or administrative procedures, EVEN IF they mention their current semester in the prompt!')
        ),
        MetadataInfo(
            name='category',
            type='str',
            description=(
                'The exact course category. You MUST use one of these exact strings: '
                '- "Βασικό κατεύθυνσης Πληροφορικής (ΒΚ-Π)", '
                '- "Κορμού (Κ)", '
                '- "Επιλογής κατευθύνσεων Πληροφορικής και Τηλεπικοινωνιών (ΕΚ-ΠΤ)", '
                '- "Επιλογής κατεύθυνσης Πληροφορικής (ΕΚ-Π)", '
                '- "Επιλογής κατεύθυνσης Τηλεπικοινωνιών (ΕΚ-Τ)", '
                '- "Ελεύθερης επιλογής (ΕΕ)", '
                '- "Βασικό κατεύθυνσης Τηλεπικοινωνιών (ΒΚ-Τ)".'
                '\nCRITICAL: DO NOT use this filter if the user asks for multiple categories (e.g. "courses for Informatics direction")'
                'Only use this filter if the user asks for ONE specific category (e.g. "I want only core courses").'
            )
        ),
    ]
)

# --- Qdrant ---
@functools.lru_cache(maxsize=1)
def get_qdrant_index() -> VectorStoreIndex:
    """Lazy loads Qdrant connection and returns the LlamaIndex VectorStoreIndex."""
    from ariadne.infrastructure.qdrant import init_qdrant_collection, get_qdrant_clients
    
    logger.info("Initializing Qdrant Vector Store...")
    init_qdrant_collection()
    client, aclient = get_qdrant_clients()
    
    qdrant_vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=settings.qdrant_collection,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
    )
    return VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store, embed_model=embed_model)

# --- Redis ---
@functools.lru_cache(maxsize=1)
def get_semantic_cache():
    """Lazy loads Redis cache. Returns (cache_index, cache_retriever). Gracefully degrades if Redis is down."""
    from ariadne.infrastructure.redis import TTLRedisVectorStore, get_redis_clients, cache_schema 
    
    logger.info("Initializing Redis Semantic Cache...")
    try:
        client, aclient = get_redis_clients()
        # Ping test to ensure it's actually alive before committing
        client.ping()
        
        cache_store = TTLRedisVectorStore(
            schema=cache_schema, 
            redis_client=client, 
            redis_aclient=aclient, 
            overwrite=False,
            ttl=settings.cache_ttl_seconds,
        )

        cache_index = VectorStoreIndex.from_vector_store(
            vector_store=cache_store, 
            embed_model=embed_model,
        )

        cache_retriever = VectorIndexRetriever(
            index=cache_index,
            similarity_top_k=1,
            embed_model=embed_model,
        )

        return cache_index, cache_retriever
    except Exception as e:
        logger.warning(f"Failed to connect to Redis. Semantic Cache is disabled. Error: {e}")
        return None, None
    