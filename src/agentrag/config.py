"""Centralized typed configuration for AgentRAG.

Reads all settings from environment variables / a .env file.
Fails loudly on missing required vars and unexpected extra vars at import time.
"""

import re
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application-wide settings backed by environment variables.

    All fields are required unless a default is provided.
    extra="forbid" ensures typos in env var names raise an error immediately
    rather than being silently ignored (SEC-009).
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="forbid",  # Fail loudly on unknown env vars (SEC-009)
    )

    # AWS credentials (optional, for local development)
    aws_access_key_id: str | None = Field(
        None,
        description="AWS access key ID.",
    )
    aws_secret_access_key: str | None = Field(
        None,
        description="AWS secret access key.",
    )

    # AWS core
    aws_region: str = Field(
        ...,
        description="AWS region, e.g. us-east-1",
        pattern=r"^[a-z]{2}-[a-z]+-\d+$",
    )

    # Bedrock (on-demand only — never provisioned throughput)
    bedrock_model_id: str = Field(
        "anthropic.claude-haiku-4-5",
        description="Bedrock model ID for LLM inference (on-demand only).",
    )
    bedrock_embedding_model_id: str = Field(
        "amazon.titan-embed-text-v2:0",
        description="Bedrock model ID for text embeddings (on-demand only).",
    )
    use_mock_embeddings: bool = Field(
        False,
        description="Whether to use mock embeddings instead of AWS Bedrock Titan.",
    )
    use_mock_llm: bool = Field(
        False,
        description="Whether to use mock LLMs instead of AWS Bedrock.",
    )

    # S3 — free tier: 5 GB storage, 20k GET, 2k PUT/month
    s3_bucket_name: str = Field(
        ...,
        description="S3 bucket for document storage and FAISS index.",
    )

    faiss_local_path: str = Field(
        "./data/index",
        description="Local directory to save/load FAISS index.",
    )

    # DynamoDB — on-demand billing, only used if M2 needs structured trace state
    dynamodb_table_name: str = Field(
        "agentrag-traces",
        description="DynamoDB table for agent run traces (on-demand billing).",
    )

    # Budget alarm threshold
    budget_alarm_threshold: float = Field(
        0.5,
        gt=0,
        description="Monthly USD threshold for the AWS Budget alert.",
    )

    # API security — minimum 32 chars enforces basic entropy (SEC-001 partial)
    api_key: str = Field(
        ...,
        min_length=32,
        description="API key for the /chat and /ingest endpoints (min 32 chars).",
    )

    @field_validator("s3_bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate S3 bucket name follows AWS naming rules."""
        if not re.match(r"^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$", v):
            raise ValueError(
                "Invalid S3 bucket name: 3-63 chars,"
                " lowercase alphanumeric/hyphens only."
            )
        return v


def get_settings() -> Settings:
    """Return a validated Settings instance.

    Fails loudly if any required var is missing or any value fails validation.
    """
    return Settings()  # type: ignore[call-arg]
