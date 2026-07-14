"""Tests for src/agentrag/config.py."""

import os
from unittest.mock import patch

import pytest

from agentrag.config import Settings, get_settings

# A cryptographically valid 32-char hex key — never use "test-key" in code (SEC-019)
_VALID_API_KEY = "a" * 32

_REQUIRED_VARS = {
    "AWS_REGION": "us-east-1",
    "S3_BUCKET_NAME": "agentrag-test-bucket-01",
    "API_KEY": _VALID_API_KEY,
}


def test_settings_loads_required_vars() -> None:
    """Settings parses required environment variables correctly."""
    with patch.dict(os.environ, _REQUIRED_VARS, clear=False):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.aws_region == "us-east-1"
    assert s.s3_bucket_name == "agentrag-test-bucket-01"
    assert s.api_key == _VALID_API_KEY


def test_settings_defaults() -> None:
    """Optional vars fall back to sensible defaults."""
    with patch.dict(os.environ, _REQUIRED_VARS, clear=False):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.bedrock_model_id == "anthropic.claude-haiku-4-5"
    assert s.budget_alarm_threshold == 0.5
    assert s.dynamodb_table_name == "agentrag-traces"


def test_settings_missing_required_raises() -> None:
    """Settings raises ValidationError when a required var is missing."""
    from pydantic import ValidationError

    env = {k: v for k, v in _REQUIRED_VARS.items() if k != "AWS_REGION"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_invalid_region_raises() -> None:
    """Settings raises ValidationError for a malformed AWS region (SEC-009)."""
    from pydantic import ValidationError

    env = {**_REQUIRED_VARS, "AWS_REGION": "INVALID_REGION"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_invalid_bucket_name_raises() -> None:
    """Settings raises ValidationError for an invalid S3 bucket name (SEC-009)."""
    from pydantic import ValidationError

    env = {**_REQUIRED_VARS, "S3_BUCKET_NAME": "INVALID BUCKET NAME!!"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_short_api_key_raises() -> None:
    """Settings raises ValidationError when API key is too short (SEC-001)."""
    from pydantic import ValidationError

    env = {**_REQUIRED_VARS, "API_KEY": "short"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_returns_settings_instance() -> None:
    """get_settings() returns a Settings object."""
    with patch.dict(os.environ, _REQUIRED_VARS, clear=False):
        s = get_settings()
    assert isinstance(s, Settings)
