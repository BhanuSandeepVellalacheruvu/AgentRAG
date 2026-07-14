# Contributing

## Commit Messages — Conventional Commits

All commits **must** follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`.

Examples:
- `feat(retrieval): add FAISS hybrid search with reranking`
- `fix(api): return 401 on missing API key`
- `docs: update ARCHITECTURE.md with M2 agent graph`

## Branch-per-Milestone Workflow

| Branch | Milestone |
|--------|-----------|
| `feature/m0-project-scaffolding` | Project scaffold |
| `feature/m1-ingestion-retrieval` | Ingestion & retrieval |
| `feature/m2-agent-orchestration` | LangGraph agent orchestration |
| `feature/m3-api-ui` | FastAPI + static UI |
| `feature/m4-aws-infrastructure` | CDK infra (Lambda, API GW, S3, DynamoDB) |
| `feature/m5-cicd` | CI/CD pipeline |
| `feature/m6-observability-evaluation` | Logging, tracing, eval |
| `feature/m7-documentation-polish` | Docs & final cleanup |

Branches are merged into `main` in order (M0 → M1 → … → M7) after review.
Never push directly to `main`.

## Standards

- CI must be **green** before any merge
- Coverage must be **≥ 80%** — no exceptions without discussion
- Run `pre-commit run --all-files` before pushing
- Every public function/class needs a docstring
- Settings go in `.env` (never hardcoded) — update `.env.example` for new vars
