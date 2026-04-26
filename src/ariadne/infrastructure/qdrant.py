import sys
import os
# Add directory to the path, so to find out all modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import logging
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from config import settings

logger = logging.getLogger("RAG_Workflow")

def get_qdrant_clients():
    """Returns synchronous and asynchronous Qdrant clients."""
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    aclient = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    return client, aclient


def init_qdrant_collection():
    """Ensures the collection and payload indexes exist before querying."""
    client, _ = get_qdrant_clients()
    
    if not client.collection_exists(settings.qdrant_collection):
        logger.info(f"Creating Qdrant collection: {settings.qdrant_collection}")
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=3072,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99, 
                    always_ram=True
                )
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2, 
                max_segment_size=100000
            )
        )
        
        # Create Payload Indexes
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="semester",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="category",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    else:
        logger.debug(f"Qdrant collection '{settings.qdrant_collection}' already exists.")