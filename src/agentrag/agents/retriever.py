"""Retriever agent for the LangGraph pipeline."""

from pathlib import Path

from agentrag.agents.state import AgentState
from agentrag.config import get_settings
from agentrag.retrieval.embeddings import (
    BedrockEmbeddingProvider,
    EmbeddingProvider,
    MockEmbeddingProvider,
)
from agentrag.retrieval.faiss_store import HybridFAISSStore

# Module-level cache for the store to avoid reloading on every lambda invocation
_STORE_INSTANCE: HybridFAISSStore | None = None


def get_store() -> HybridFAISSStore:
    """Get or initialize the FAISS store."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is not None:
        return _STORE_INSTANCE

    settings = get_settings()
    provider: EmbeddingProvider
    if settings.use_mock_embeddings:
        provider = MockEmbeddingProvider(dimension=1024)
    else:
        provider = BedrockEmbeddingProvider(
            model_id=settings.bedrock_embedding_model_id,
            aws_region=settings.aws_region,
        )
    store = HybridFAISSStore(
        embedding_provider=provider,
        dimension=provider.get_dimension(),
        persist_dir=Path(settings.faiss_local_path),
    )

    if settings.s3_bucket_name:
        try:
            store.load_from_s3(settings.s3_bucket_name)
        except Exception:
            # Fallback to local if S3 fails or is empty
            store.load_local()
    else:
        store.load_local()

    _STORE_INSTANCE = store
    return _STORE_INSTANCE


def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve documents using the query."""
    settings = get_settings()
    store = get_store()
    alpha = 0.0 if settings.use_mock_embeddings else 0.5
    results = store.search(state["query"], top_k=3, alpha=alpha)

    return {**state, "context": results, "steps": ["retrieved_context"]}
