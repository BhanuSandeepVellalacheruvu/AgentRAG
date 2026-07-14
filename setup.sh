#!/bin/bash
set -e

# 1. Package structure
mkdir -p src/agentrag/ingestion
mkdir -p src/agentrag/retrieval
mkdir -p src/agentrag/agents
mkdir -p src/agentrag/api
mkdir -p src/agentrag/infra
mkdir -p src/agentrag/common
touch src/agentrag/__init__.py
touch src/agentrag/ingestion/__init__.py
touch src/agentrag/retrieval/__init__.py
touch src/agentrag/agents/__init__.py
touch src/agentrag/api/__init__.py
touch src/agentrag/infra/__init__.py
touch src/agentrag/common/__init__.py

mkdir -p tests
touch tests/__init__.py

# 2. Tooling Configs
cat << 'INNER_EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "agentrag"
version = "0.1.0"
description = "A multi-agent RAG application"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic-settings>=2.0.0",
    "pydantic>=2.0.0"
]

[project.optional-dependencies]
dev = [
    "ruff",
    "mypy",
    "pytest",
    "pytest-cov",
    "pre-commit",
    "pip-tools"
]
test = [
    "pytest",
    "pytest-cov"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP"]
ignore = []

[tool.mypy]
strict = true
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
addopts = "--cov=src/agentrag --cov-fail-under=80"
testpaths = ["tests"]
INNER_EOF

cat << 'INNER_EOF' > .ruff.toml
[lint]
select = ["E", "F", "I", "W", "UP"]
INNER_EOF

cat << 'INNER_EOF' > mypy.ini
[mypy]
strict = true
python_version = 3.11
warn_return_any = true
warn_unused_configs = true
INNER_EOF

cat << 'INNER_EOF' > .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.8
    hooks:
    -   id: ruff
        args: [ --fix ]
    -   id: ruff-format
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
    -   id: mypy
        additional_dependencies: [pydantic]
INNER_EOF

cat << 'INNER_EOF' > .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run Ruff
      run: ruff check src/ tests/
    - name: Run Mypy
      run: mypy src/ tests/
    - name: Run Pytest with Coverage
      run: pytest --cov=src/agentrag --cov-fail-under=80
INNER_EOF

# 3. .env.example
cat << 'INNER_EOF' > .env.example
# AWS Region
AWS_REGION=us-east-1

# Bedrock
BEDROCK_MODEL_ID=amazon.titan-embed-text-v1

# AWS Infrastructure
S3_BUCKET_NAME=agentrag-docs-dummy
DYNAMODB_TABLE_NAME=agentrag-traces-dummy
BUDGET_ALARM_THRESHOLD=2.0
INNER_EOF

# 4. Config loader
cat << 'INNER_EOF' > src/agentrag/common/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    aws_region: str
    bedrock_model_id: str
    s3_bucket_name: str
    dynamodb_table_name: str
    budget_alarm_threshold: float

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_settings() -> Settings:
    return Settings() # type: ignore
INNER_EOF

# 5. Tests
cat << 'INNER_EOF' > tests/test_config.py
import os
from agentrag.common.config import get_settings

def test_get_settings_loads_env() -> None:
    os.environ["AWS_REGION"] = "us-west-2"
    os.environ["BEDROCK_MODEL_ID"] = "dummy-model"
    os.environ["S3_BUCKET_NAME"] = "test-bucket"
    os.environ["DYNAMODB_TABLE_NAME"] = "test-table"
    os.environ["BUDGET_ALARM_THRESHOLD"] = "2.0"
    
    settings = get_settings()
    assert settings.aws_region == "us-west-2"
    assert settings.bedrock_model_id == "dummy-model"
    assert settings.s3_bucket_name == "test-bucket"
    assert settings.dynamodb_table_name == "test-table"
    assert settings.budget_alarm_threshold == 2.0
INNER_EOF

# 6. Documentation
cat << 'INNER_EOF' > README.md
# AgentRAG

A multi-agent RAG application relying entirely on zero-cost AWS infrastructure.

## Architecture

*Placeholder: To be filled in subsequent milestones.*
INNER_EOF

cat << 'INNER_EOF' > CONTRIBUTING.md
# Contributing

## Conventional Commits
All commits must follow [Conventional Commits](https://www.conventionalcommits.org/).

## Branch Workflow
We use a branch-per-milestone workflow. Each feature is developed on a separate branch (e.g. `feature/m0-project-scaffolding`) and merged sequentially into `main`.
INNER_EOF

