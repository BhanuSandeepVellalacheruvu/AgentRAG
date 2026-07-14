"""Tests for tracing observability module."""

from unittest.mock import MagicMock, patch

from agentrag.observability.tracing import log_trace


@patch("agentrag.observability.tracing.boto3")
@patch("agentrag.observability.tracing.get_settings")
def test_log_trace_success(mock_get_settings: MagicMock, mock_boto3: MagicMock) -> None:
    """log_trace successfully calls boto3 put_item."""
    mock_settings = MagicMock()
    mock_settings.dynamodb_table_name = "my-test-table"
    mock_settings.aws_region = "us-east-1"
    mock_settings.aws_access_key_id = "test-key"
    mock_settings.aws_secret_access_key = "test-secret"
    mock_get_settings.return_value = mock_settings

    mock_dynamodb = MagicMock()
    mock_table = MagicMock()
    mock_boto3.resource.return_value = mock_dynamodb
    mock_dynamodb.Table.return_value = mock_table

    log_trace(
        query="what is the holiday policy?",
        generation="12 days of holidays.",
        steps=["routed_to_retrieve", "retrieved_context"],
        grounded=True,
        latency_ms=120.5,
        session_id="session-123",
    )

    mock_boto3.resource.assert_called_once_with(
        "dynamodb",
        region_name="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )
    mock_dynamodb.Table.assert_called_once_with("my-test-table")

    # Verify put_item was called with expected fields
    _, call_kwargs = mock_table.put_item.call_args
    item = call_kwargs["Item"]
    assert item["query"] == "what is the holiday policy?"
    assert item["reply"] == "12 days of holidays."
    assert item["steps"] == ["routed_to_retrieve", "retrieved_context"]
    assert item["grounded"] is True
    assert item["latency_ms"] == 120.5
    assert item["session_id"] == "session-123"
    assert "id" in item
    assert "timestamp" in item


@patch("agentrag.observability.tracing.boto3")
@patch("agentrag.observability.tracing.get_settings")
def test_log_trace_handles_exception_gracefully(
    mock_get_settings: MagicMock, mock_boto3: MagicMock
) -> None:
    """log_trace handles boto3 exceptions gracefully without crashing."""
    mock_settings = MagicMock()
    mock_settings.dynamodb_table_name = "my-test-table"
    mock_get_settings.return_value = mock_settings

    # Simulate connection error
    mock_boto3.resource.side_effect = Exception("AWS Connection Error")

    # Should not raise exception
    log_trace(
        query="test query",
        generation="test generation",
        steps=["step"],
        grounded=True,
        latency_ms=10.0,
    )
    mock_boto3.resource.assert_called_once()
