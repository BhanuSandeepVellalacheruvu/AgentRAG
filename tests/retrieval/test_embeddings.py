"""Tests for src/agentrag/retrieval/embeddings.py."""

import json
from unittest.mock import MagicMock, patch

from agentrag.retrieval.embeddings import BedrockEmbeddingProvider, MockEmbeddingProvider


def test_mock_embedding_provider() -> None:
    """Test MockEmbeddingProvider returns expected dimensions and handles empty strings."""
    provider = MockEmbeddingProvider(dimension=5)
    
    texts = ["hello", "world", "  "]
    embeddings = provider.embed_texts(texts)
    
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 5
    assert len(embeddings[1]) == 5
    assert len(embeddings[2]) == 5
    
    # Empty string should be all zeros
    assert all(x == 0.0 for x in embeddings[2])
    
    # Non-empty should be normalized
    norm = sum(x*x for x in embeddings[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6


@patch("boto3.client")
def test_bedrock_embedding_provider(mock_boto3_client: MagicMock) -> None:
    """Test BedrockEmbeddingProvider invokes the AWS client correctly."""
    mock_client = MagicMock()
    mock_boto3_client.return_value = mock_client
    
    # Setup mock response
    mock_response = {
        "body": MagicMock(read=MagicMock(return_value=json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode("utf-8")))
    }
    mock_client.invoke_model.return_value = mock_response
    
    provider = BedrockEmbeddingProvider(model_id="test-model", aws_region="us-east-1")
    
    texts = ["test1", "  ", "test3"]
    embeddings = provider.embed_texts(texts)
    
    assert len(embeddings) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[2] == [0.1, 0.2, 0.3]
    
    # Empty string is handled without API call
    assert len(embeddings[1]) == 1024
    assert all(x == 0.0 for x in embeddings[1])
    
    # Should have been called twice (for the two non-empty strings)
    assert mock_client.invoke_model.call_count == 2
    
    # Verify call args
    call_args = mock_client.invoke_model.call_args_list[0][1]
    assert call_args["modelId"] == "test-model"
    assert "test1" in call_args["body"]
