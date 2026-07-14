# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-07-14

### Added
- `src/agentrag/` src-layout package with sub-packages:
  `agents/`, `retrieval/`, `tools/`, `api/`, `infra/`, `config.py`
- `tests/` mirroring the package structure
- `pyproject.toml` (PEP 621) with `uv` as the package/lockfile manager
- Tooling: `ruff` (lint + format), `mypy` (strict-ish), `pytest` + `pytest-cov`
- `pre-commit` hooks wiring ruff, mypy, and standard checks
- GitHub Actions CI workflow (`ci.yml`) — hard-fails on lint/type/test/coverage < 80%
- `.env.example` documenting all config vars across M0–M6
- Typed settings loader via `pydantic-settings` (`src/agentrag/config.py`)
- `.gitignore` for Python / CDK / Node / macOS
- MIT `LICENSE`
- `README.md` with project overview, local setup, and cost profile
- `ARCHITECTURE.md` stub
- `CONTRIBUTING.md` with Conventional Commits and branch-per-milestone workflow
