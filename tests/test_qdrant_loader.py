import pytest
from unittest.mock import MagicMock, patch
from pipelines.loaders.qdrant_loader import QdrantLoader
from llama_index.core.schema import TextNode

@patch("pipelines.loaders.qdrant_loader.VectorStoreIndex")
def test_qdrant_loader_batching(mock_index):
    storage_mock = MagicMock()
    mock_index_instance = MagicMock()
    mock_index.return_value = mock_index_instance
    
    loader = QdrantLoader(storage_context=storage_mock, embed_model=MagicMock())
    nodes = [TextNode(text=f"Valid text {i}") for i in range(50)]
    
    loader.load_nodes(nodes, batch_size=20)
    assert mock_index_instance.insert_nodes.call_count == 3

@patch("pipelines.loaders.qdrant_loader.time.sleep", return_value=None)
@patch("pipelines.loaders.qdrant_loader.VectorStoreIndex")
def test_qdrant_loader_retry_success(mock_index, mock_sleep):
    storage_mock = MagicMock()
    mock_index_instance = MagicMock()
    mock_index_instance.insert_nodes.side_effect = [Exception("503 Service Unavailable"), None]
    mock_index.return_value = mock_index_instance
    
    loader = QdrantLoader(storage_context=storage_mock, max_retries=3, embed_model=MagicMock())
    nodes = [TextNode(text="Valid text")]
    
    loader.load_nodes(nodes, batch_size=20)
    assert mock_index_instance.insert_nodes.call_count == 2
