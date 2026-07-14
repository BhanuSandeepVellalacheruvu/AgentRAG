"""Retrieval module for AgentRAG.

Provides document ingestion, chunking, embedding, and hybrid vector search capabilities.
"""

from agentrag.retrieval.chunker import Chunk, Chunker
from agentrag.retrieval.embeddings import (
    BedrockEmbeddingProvider,
    EmbeddingProvider,
    MockEmbeddingProvider,
)
from agentrag.retrieval.faiss_store import HybridFAISSStore, SearchResult
from agentrag.retrieval.loader import Document, DocumentLoader

__all__ = [
    "Chunk",
    "Chunker",
    "BedrockEmbeddingProvider",
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "HybridFAISSStore",
    "SearchResult",
    "Document",
    "DocumentLoader",
]
