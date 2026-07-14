"""Embeddings generation module.

Provides a protocol for embedding providers and implementations for AWS Bedrock
and a local mock for testing.
"""

import json
from typing import Protocol

import boto3


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...


class BedrockEmbeddingProvider:
    """Generates embeddings using Amazon Bedrock Titan."""

    def __init__(self, model_id: str, aws_region: str | None = None) -> None:
        """Initialize the Bedrock provider.

        Args:
            model_id: The Bedrock model ID (e.g., 'amazon.titan-embed-text-v2:0').
            aws_region: AWS region.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=aws_region)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings by calling Bedrock one text at a time.

        Amazon Titan text embedding models typically only accept one string per request.
        """
        embeddings = []
        for text in texts:
            if not text.strip():
                # Provide a zero vector or handle appropriately.
                # Assuming 1024 dims for Titan V2
                embeddings.append([0.0] * 1024)
                continue

            body = json.dumps({"inputText": text})
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embeddings.append(response_body.get("embedding", []))

        return embeddings


class MockEmbeddingProvider:
    """A mock embedding provider for CI and testing that generates pseudo-random vectors."""  # noqa: E501

    def __init__(self, dimension: int = 1024) -> None:
        """Initialize mock provider.

        Args:
            dimension: Dimensionality of the mock embeddings.
        """
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic fake embeddings based on text length."""
        embeddings = []
        for text in texts:
            if not text.strip():
                embeddings.append([0.0] * self.dimension)
                continue

            # Simple deterministic but non-zero generation
            length = len(text)
            vec = [(float(i + length) % 10.0) / 10.0 for i in range(self.dimension)]
            # Normalize it roughly so FAISS cosine sim makes sense
            norm = sum(x * x for x in vec) ** 0.5
            vec = [x / (norm or 1.0) for x in vec]
            embeddings.append(vec)

        return embeddings
