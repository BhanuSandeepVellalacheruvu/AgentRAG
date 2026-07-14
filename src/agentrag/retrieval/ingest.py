"""CLI tool to ingest documents and build the FAISS index."""

import argparse
import sys
from pathlib import Path

from agentrag.config import get_settings
from agentrag.retrieval.chunker import Chunker
from agentrag.retrieval.embeddings import (
    BedrockEmbeddingProvider,
    EmbeddingProvider,
    MockEmbeddingProvider,
)
from agentrag.retrieval.faiss_store import HybridFAISSStore
from agentrag.retrieval.loader import DocumentLoader


def main() -> None:
    """Run the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into AgentRAG FAISS index."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/bhanusandeepvellalacheruvu/Documents/JOB/hr-policies-qa-dataset",
        help="Path to the directory containing documents to ingest.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embeddings instead of AWS Bedrock Titan.",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Resolve argument name fallback due to argparse hyphens
    data_dir_path = Path(getattr(args, "data_dir", args.data_dir))

    if not data_dir_path.exists():
        print(
            f"Error: Data directory '{data_dir_path}' does not exist.", file=sys.stderr
        )
        sys.exit(1)

    print(f"Loading documents from {data_dir_path}...")
    loader = DocumentLoader()
    documents = list(loader.load_local_directory(data_dir_path))
    print(f"Loaded {len(documents)} document turns.")

    print("Chunking documents...")
    chunker = Chunker(chunk_size=400, chunk_overlap=80)
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embedding provider...")
    provider: EmbeddingProvider
    if args.mock:
        print("Using Mock Embedding Provider (local only, no AWS charges).")
        provider = MockEmbeddingProvider(dimension=1024)
    else:
        print(
            f"Using AWS Bedrock Titan Embeddings (Model: {settings.bedrock_embedding_model_id})"  # noqa: E501
        )
        provider = BedrockEmbeddingProvider(
            model_id=settings.bedrock_embedding_model_id,
            aws_region=settings.aws_region,
        )

    print(f"Building FAISS store at {settings.faiss_local_path}...")
    store = HybridFAISSStore(
        embedding_provider=provider,
        dimension=provider.get_dimension(),
        persist_dir=settings.faiss_local_path,
    )

    store.add_chunks(chunks)
    store.save_local()
    print("Ingestion complete! Local index built successfully.")


if __name__ == "__main__":
    main()
