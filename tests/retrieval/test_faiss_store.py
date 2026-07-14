"""Tests for src/agentrag/retrieval/faiss_store.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import hashlib

import numpy as np
import pytest

from agentrag.retrieval.chunker import Chunk
from agentrag.retrieval.embeddings import MockEmbeddingProvider
from agentrag.retrieval.faiss_store import HybridFAISSStore


@pytest.fixture
def temp_store_dir(tmp_path: Path) -> Path:
    return tmp_path / "index"


def test_add_chunks_and_search(temp_store_dir: Path) -> None:
    """Test adding chunks and searching hybrid index."""
    provider = MockEmbeddingProvider(dimension=5)
    store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    
    chunks = [
        Chunk(text="apple banana", filename="1.txt", source="1.txt", chunk_index=0),
        Chunk(text="orange grapes", filename="2.txt", source="2.txt", chunk_index=0),
        Chunk(text="apple orange kiwi", filename="3.txt", source="3.txt", chunk_index=0),
    ]
    
    store.add_chunks(chunks)
    assert len(store.chunks) == 3
    assert store.index.ntotal == 3
    
    # Pure sparse search to avoid mock embedding artifacts
    res = store.search("apple", top_k=2, alpha=0.0)
    assert len(res) == 2
    # Ensure apple documents rank higher
    # "apple banana" and "apple orange" both have 'apple' for BM25
    assert res[0].chunk.filename in ["1.txt", "3.txt"]
    assert res[1].chunk.filename in ["1.txt", "3.txt"]
    
    # Test empty chunks array
    empty_store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir.parent / "empty")
    assert empty_store.search("test") == []


def test_save_and_load_local(temp_store_dir: Path) -> None:
    """Test saving and loading the index locally."""
    provider = MockEmbeddingProvider(dimension=5)
    store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    
    chunks = [Chunk(text="test text", filename="1.txt", source="1.txt", chunk_index=0)]
    store.add_chunks(chunks)
    store.save_local()
    
    assert (temp_store_dir / "index.faiss").exists()
    assert (temp_store_dir / "metadata.json").exists()
    
    # Load into new store
    new_store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    new_store.load_local()
    
    assert new_store.index.ntotal == 1
    assert len(new_store.chunks) == 1
    assert new_store.chunks[0].text == "test text"


def test_load_local_not_found(temp_store_dir: Path) -> None:
    """Test loading non-existent index raises FileNotFoundError."""
    provider = MockEmbeddingProvider(dimension=5)
    store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    with pytest.raises(FileNotFoundError):
        store.load_local()


@patch("boto3.client")
def test_sync_to_s3(mock_boto3_client: MagicMock, temp_store_dir: Path) -> None:
    """Test syncing index to S3 and DynamoDB."""
    mock_s3 = MagicMock()
    mock_ddb = MagicMock()
    def get_client(service, **kwargs):
        if service == "s3": return mock_s3
        if service == "dynamodb": return mock_ddb
    mock_boto3_client.side_effect = get_client
    
    provider = MockEmbeddingProvider(dimension=5)
    store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    store.add_chunks([Chunk(text="t", filename="1.txt", source="1", chunk_index=0)])
    
    store.sync_to_s3("my-bucket", prefix="idx/")
    
    assert mock_s3.put_object.call_count == 2
    assert mock_ddb.put_item.call_count == 1
    
    call_args = mock_ddb.put_item.call_args[1]
    assert call_args["TableName"] == "agentrag-index-metadata"
    assert call_args["Item"]["index_id"]["S"] == "my-bucket/idx/index.faiss"


@patch("boto3.client")
def test_load_from_s3_success(mock_boto3_client: MagicMock, temp_store_dir: Path) -> None:
    """Test loading from S3 with successful integrity check."""
    # First create a local index to generate valid faiss data
    provider = MockEmbeddingProvider(dimension=5)
    store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    store.add_chunks([Chunk(text="t", filename="1.txt", source="1", chunk_index=0)])
    store.save_local()
    
    with open(temp_store_dir / "index.faiss", "rb") as f:
        faiss_data = f.read()
    with open(temp_store_dir / "metadata.json", "rb") as f:
        meta_data = f.read()
        
    valid_hash = hashlib.sha256(faiss_data).hexdigest()
    
    mock_s3 = MagicMock()
    mock_ddb = MagicMock()
    def get_client(service, **kwargs):
        if service == "s3": return mock_s3
        if service == "dynamodb": return mock_ddb
    mock_boto3_client.side_effect = get_client
    
    def get_object_side_effect(Bucket, Key):
        if "faiss" in Key:
            return {"Body": MagicMock(read=MagicMock(return_value=faiss_data))}
        return {"Body": MagicMock(read=MagicMock(return_value=meta_data))}
    mock_s3.get_object.side_effect = get_object_side_effect
    
    mock_ddb.get_item.return_value = {
        "Item": {"sha256_hash": {"S": valid_hash}}
    }
    
    # Load into new dir
    new_dir = temp_store_dir.parent / "new_store"
    new_store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=new_dir)
    new_store.load_from_s3("bucket")
    
    assert new_store.index.ntotal == 1


@patch("boto3.client")
def test_load_from_s3_integrity_failure(mock_boto3_client: MagicMock, temp_store_dir: Path) -> None:
    """Test loading from S3 fails when integrity check fails."""
    mock_s3 = MagicMock()
    mock_ddb = MagicMock()
    def get_client(service, **kwargs):
        if service == "s3": return mock_s3
        if service == "dynamodb": return mock_ddb
    mock_boto3_client.side_effect = get_client
    
    mock_s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"fake data"))}
    mock_ddb.get_item.return_value = {
        "Item": {"sha256_hash": {"S": "wrong_hash"}}
    }
    
    provider = MockEmbeddingProvider(dimension=5)
    store = HybridFAISSStore(embedding_provider=provider, dimension=5, persist_dir=temp_store_dir)
    
    with pytest.raises(RuntimeError, match="integrity check failed"):
        store.load_from_s3("bucket")
