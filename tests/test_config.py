"""Tests for src/agentrag/config.py."""

import os
from unittest.mock import patch

import pytest

from agentrag.config import Settings, get_settings

_REQUIRED_VARS = {
    "AWS_REGION": "us-east-1",
    "S3_BUCKET_NAME": "test-bucket",
    "API_KEY": "test-key",
}


def test_settings_loads_required_vars() -> None:
    """Settings parses required environment variables correctly."""
    with patch.dict(os.environ, _REQUIRED_VARS, clear=False):
        s = Settings()  # type: ignore[call-arg]
    assert s.aws_region == "us-east-1"
    assert s.s3_bucket_name == "test-bucket"
    assert s.api_key == "test-key"


def test_settings_defaults() -> None:
    """Optional vars fall back to sensible defaults."""
    with patch.dict(os.environ, _REQUIRED_VARS, clear=False):
        s = Settings()  # type: ignore[call-arg]
    assert s.bedrock_model_id == "anthropic.claude-haiku-4-5"
    assert s.budget_alarm_threshold == 2.0
    assert s.dynamodb_table_name == "agentrag-traces"


def test_settings_missing_required_raises() -> None:
    """Settings raises ValidationError when a required var is missing."""
    from pydantic import ValidationError

    env = {k: v for k, v in _REQUIRED_VARS.items() if k != "AWS_REGION"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]


def test_get_settings_returns_settings_instance() -> None:
    """get_settings() returns a Settings object."""
    with patch.dict(os.environ, _REQUIRED_VARS, clear=False):
        s = get_settings()
    assert isinstance(s, Settings)
