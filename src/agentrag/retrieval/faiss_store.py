"""FAISS vector store and hybrid retrieval module.

Provides a persistent FAISS index with BM25 hybrid search capabilities
and S3 integration for remote storage and integrity validation.
"""

import hashlib
import json
import logging
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

import boto3
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from agentrag.retrieval.chunker import Chunk
from agentrag.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk with a relevance score."""

    chunk: Chunk
    score: float


class HybridFAISSStore:
    """A hybrid retriever using FAISS (dense) and BM25 (sparse)."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        dimension: int = 1024,
        persist_dir: str | Path = "./data/index",
    ) -> None:
        """Initialize the store.

        Args:
            embedding_provider: The provider to generate embeddings.
            dimension: Dimensionality of the embeddings.
            persist_dir: Local directory to save/load index files.
        """
        self.embedding_provider = embedding_provider
        self.dimension = dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Inner product index (equivalent to cosine similarity if vectors are normalized)  # noqa: E501
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Embed and add chunks to the index."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedding_provider.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)

        vectors = np.array(all_embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

        # Rebuild BM25
        tokenized_corpus = [text.lower().split() for text in texts]
        if self.bm25:
            # We must rebuild from all chunks to maintain term frequencies correctly
            all_texts = [c.text for c in self.chunks]
            all_tokenized = [t.lower().split() for t in all_texts]
            self.bm25 = BM25Okapi(all_tokenized)
        else:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[SearchResult]:  # noqa: E501
        """Perform hybrid search combining FAISS and BM25.

        Args:
            query: The search query.
            top_k: Number of results to return.
            alpha: Weight for dense (FAISS) vs sparse (BM25) search.
                   1.0 = dense only, 0.0 = sparse only.
        """
        if not self.chunks:
            return []

        # 1. Dense Search (FAISS)
        query_vector = np.array(
            self.embedding_provider.embed_texts([query]), dtype=np.float32
        )  # noqa: E501
        faiss.normalize_L2(query_vector)

        # Get more results than top_k for better fusion
        search_k = min(top_k * 2, len(self.chunks))
        dense_scores, dense_indices = self.index.search(query_vector, search_k)

        # 2. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = (
            self.bm25.get_scores(tokenized_query)
            if self.bm25
            else [0.0] * len(self.chunks)
        )  # noqa: E501

        # Normalize scores to [0, 1] for fusion
        def min_max_norm(scores: list[float] | np.ndarray) -> np.ndarray:
            s = np.array(scores)
            min_s, max_s = np.min(s), np.max(s)
            if max_s - min_s == 0:
                return np.ones_like(s) if max_s > 0 else np.zeros_like(s)
            return (s - min_s) / (max_s - min_s)

        # Get all dense scores mapped to original indices (default 0.0)
        all_dense_scores = np.zeros(len(self.chunks))
        for idx, score in zip(dense_indices[0], dense_scores[0]):
            if idx != -1:
                all_dense_scores[idx] = score

        norm_dense = min_max_norm(all_dense_scores)
        norm_sparse = min_max_norm(sparse_scores)

        # 3. Score Fusion
        hybrid_scores = alpha * norm_dense + (1.0 - alpha) * norm_sparse

        # 4. Rank and return top_k
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if hybrid_scores[idx] > 0.0:  # Only return relevant results
                results.append(
                    SearchResult(
                        chunk=self.chunks[idx], score=float(hybrid_scores[idx])
                    )
                )  # noqa: E501

        return results

    def save_local(self) -> None:
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))

        metadata = [asdict(chunk) for chunk in self.chunks]
        with open(self.persist_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def load_local(self) -> None:
        """Load the FAISS index and metadata from disk."""
        index_path = self.persist_dir / "index.faiss"
        meta_path = self.persist_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Index files not found")

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self.chunks = [Chunk(**data) for data in metadata]

        # Rebuild BM25
        if self.chunks:
            all_tokenized = [c.text.lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(all_tokenized)

    def sync_to_s3(
        self, s3_bucket: str, prefix: str = "index/", aws_region: str | None = None
    ) -> None:  # noqa: E501
        """Upload index files to S3 and store hash in DynamoDB for integrity."""
        self.save_local()

        s3 = boto3.client("s3", region_name=aws_region)
        dynamodb = boto3.client("dynamodb", region_name=aws_region)

        index_path = self.persist_dir / "index.faiss"
        with open(index_path, "rb") as f:
            index_data = f.read()

        # SEC-003: Calculate SHA-256 hash for integrity
        index_hash = hashlib.sha256(index_data).hexdigest()

        # Upload files
        s3.put_object(Bucket=s3_bucket, Key=f"{prefix}index.faiss", Body=index_data)

        with open(self.persist_dir / "metadata.json", "rb") as f:
            s3.put_object(Bucket=s3_bucket, Key=f"{prefix}metadata.json", Body=f.read())

        # Store hash in DynamoDB (SEC-003)
        try:
            dynamodb.put_item(
                TableName="agentrag-index-metadata",
                Item={
                    "index_id": {"S": f"{s3_bucket}/{prefix}index.faiss"},
                    "sha256_hash": {"S": index_hash},
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store index hash in DynamoDB: {e}")

    def load_from_s3(
        self, s3_bucket: str, prefix: str = "index/", aws_region: str | None = None
    ) -> None:  # noqa: E501
        """Download index from S3, verify integrity, and load."""
        s3 = boto3.client("s3", region_name=aws_region)
        dynamodb = boto3.client("dynamodb", region_name=aws_region)

        # Download index
        response = s3.get_object(Bucket=s3_bucket, Key=f"{prefix}index.faiss")
        index_data = response["Body"].read()

        # SEC-003: Verify integrity
        try:
            db_res = dynamodb.get_item(
                TableName="agentrag-index-metadata",
                Key={"index_id": {"S": f"{s3_bucket}/{prefix}index.faiss"}},
            )
            if "Item" in db_res:
                expected_hash = db_res["Item"]["sha256_hash"]["S"]
                actual_hash = hashlib.sha256(index_data).hexdigest()
                if not secrets.compare_digest(actual_hash, expected_hash):
                    raise RuntimeError(
                        "SecurityError: FAISS index integrity check failed — possible tampering"  # noqa: E501
                    )  # noqa: E501
        except Exception as e:
            if "SecurityError" in str(e):
                raise
            logger.warning(f"Could not verify index integrity via DynamoDB: {e}")

        # Save downloaded data and load
        with open(self.persist_dir / "index.faiss", "wb") as f:
            f.write(index_data)

        meta_res = s3.get_object(Bucket=s3_bucket, Key=f"{prefix}metadata.json")
        with open(self.persist_dir / "metadata.json", "wb") as f:
            f.write(meta_res["Body"].read())

        self.load_local()
