"""Centralized typed configuration for AgentRAG.

Reads all settings from environment variables / a .env file.
Fails loudly on missing required vars at import time via pydantic-settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AWS core
    aws_region: str = Field(..., description="AWS region, e.g. us-east-1")

    # Bedrock
    bedrock_model_id: str = Field(
        "anthropic.claude-haiku-4-5",
        description="Bedrock model ID for LLM inference (on-demand only).",
    )
    bedrock_embedding_model_id: str = Field(
        "amazon.titan-embed-text-v2:0",
        description="Bedrock model ID for text embeddings (on-demand only).",
    )

    # S3
    s3_bucket_name: str = Field(
        ..., description="S3 bucket for document storage and FAISS index."
    )

    # DynamoDB (optional — only used if M2 needs structured trace state)
    dynamodb_table_name: str = Field(
        "agentrag-traces",
        description="DynamoDB table for agent run traces (on-demand billing).",
    )

    # Budget
    budget_alarm_threshold: float = Field(
        2.0, description="Monthly USD threshold for the AWS Budget alert."
    )

    # API security
    api_key: str = Field(
        ..., description="API key for the /chat and /ingest endpoints."
    )


def get_settings() -> Settings:
    """Return a Settings instance, failing loudly on missing required vars."""
    return Settings()  # type: ignore[call-arg]
