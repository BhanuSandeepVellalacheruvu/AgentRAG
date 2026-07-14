"""Observability tracing for AgentRAG."""

import logging
import time
import uuid
from typing import Any

import boto3

from agentrag.config import get_settings

logger = logging.getLogger("agentrag.observability")


def log_trace(
    query: str,
    generation: str,
    steps: list[str],
    grounded: bool,
    latency_ms: float,
    session_id: str | None = None,
) -> None:
    """Save an execution trace to the DynamoDB traces table."""
    settings = get_settings()
    table_name = settings.dynamodb_table_name

    # Construct the trace record
    trace_id = str(uuid.uuid4())
    timestamp = float(time.time())

    item: dict[str, Any] = {
        "id": trace_id,
        "timestamp": timestamp,
        "query": query,
        "reply": generation,
        "steps": steps,
        "grounded": grounded,
        "latency_ms": latency_ms,
    }

    if session_id:
        item["session_id"] = session_id

    try:
        # Initialize DynamoDB resource
        # Boto3 will automatically use AWS credentials from settings/environment
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        table = dynamodb.Table(table_name)
        table.put_item(Item=item)
        logger.info(f"Trace {trace_id} successfully saved to DynamoDB.")
    except Exception as e:
        # Gracefully handle failures to ensure API resilience
        logger.error(f"Failed to write trace to DynamoDB: {e}")
