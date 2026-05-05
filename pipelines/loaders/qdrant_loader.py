import time
import sys
import os
from typing import List

from llama_index.core.schema import BaseNode
from llama_index.core import VectorStoreIndex, Settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipelines.core.logger import get_logger

logger = get_logger(__name__)

class QdrantLoader:
    def __init__(self, storage_context, embed_model=None, max_retries: int = 3):
        self.storage_context = storage_context
        self.embed_model = embed_model or Settings.embed_model
        self.max_retries = max_retries

        self.index = VectorStoreIndex(
            nodes=[],
            embed_model=self.embed_model,
            storage_context=self.storage_context,
            show_progress=True,
        )

    def load_nodes(self, nodes: List[BaseNode], batch_size: int = 20):
        valid_nodes = [node for node in nodes if node.get_content() and len(node.get_content().strip()) > 5]
        
        logger.info(f"Starting robust ingestion for {len(valid_nodes)} valid nodes...")
        successful_nodes = 0
        failed_nodes = 0

        for i in range(0, len(valid_nodes), batch_size):
            batch = valid_nodes[i:i + batch_size]
            success = False

            for attempt in range(self.max_retries):
                try:
                    self.index.insert_nodes(batch)
                    success = True
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timed out" in error_msg or "503" in error_msg or "500" in error_msg:
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Timeout/Server error on batch {i//batch_size}. Retrying ({attempt + 1}/{self.max_retries}) in 2 secs...")
                            time.sleep(2)
                        else:
                            logger.error(f"Permanent failure for batch {i//batch_size} after {self.max_retries} attempts: {e}")
                    else:
                        logger.error(f"Skipping problematic batch {i//batch_size}. Error: {e}")
                        break

            if success:
                successful_nodes += len(batch)
                logger.info(f"Successfully integrated {successful_nodes}/{len(valid_nodes)} nodes...")
            else:
                failed_nodes += len(batch)

        logger.info("=" * 50)
        logger.info("🎉 Ingestion Completed!")
        logger.info(f"Successes: {successful_nodes}")
        logger.info(f"Failures: {failed_nodes}")
        logger.info("=" * 50)
