"""Tests for src/agentrag/agents/retriever.py."""

from unittest.mock import ANY, MagicMock, patch

from agentrag.agents.retriever import get_store, retrieve_context
from agentrag.retrieval.chunker import Chunk
from agentrag.retrieval.faiss_store import SearchResult


@patch("agentrag.agents.retriever.get_settings")
@patch("agentrag.agents.retriever.BedrockEmbeddingProvider")
@patch("agentrag.agents.retriever.HybridFAISSStore")
def test_get_store_loads_s3(
    mock_store_cls: MagicMock, mock_provider_cls: MagicMock, mock_settings: MagicMock
) -> None:
    """Test get_store initializes and loads from S3."""
    import agentrag.agents.retriever as ret_mod

    # Reset cache
    ret_mod._STORE_INSTANCE = None

    mock_settings_obj = MagicMock()
    mock_settings_obj.aws_region = "us-east-1"
    mock_settings_obj.faiss_local_path = "/tmp"
    mock_settings_obj.s3_bucket_name = "my-bucket"
    mock_settings.return_value = mock_settings_obj

    mock_store_instance = MagicMock()
    mock_store_cls.return_value = mock_store_instance

    store = get_store()

    assert store == mock_store_instance
    mock_store_instance.load_from_s3.assert_called_once_with("my-bucket")

    # Check cache works
    store2 = get_store()
    assert store2 == mock_store_instance
    assert mock_store_cls.call_count == 1

    # Reset cache for other tests
    ret_mod._STORE_INSTANCE = None


@patch("agentrag.agents.retriever.get_store")
def test_retrieve_context(mock_get_store: MagicMock) -> None:
    """Test retrieve_context calls search."""
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store

    expected_result = [
        SearchResult(
            chunk=Chunk(text="test", filename="a.txt", source="a", chunk_index=0),
            score=1.0,
        )
    ]
    mock_store.search.return_value = expected_result

    state = {"query": "test query"}
    new_state = retrieve_context(state)  # type: ignore

    mock_store.search.assert_called_once_with("test query", top_k=3, alpha=ANY)
    assert new_state["context"] == expected_result
    assert new_state["steps"] == ["retrieved_context"]
