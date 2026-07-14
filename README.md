# AgentRAG

> A production-grade multi-agent Retrieval-Augmented Generation (RAG) system
> deployed on zero-cost AWS infrastructure.

## What Is This?

AgentRAG is a multi-agent AI pipeline that answers questions by intelligently
routing queries through retrieval, tool-calling, and critique agents — all
orchestrated with LangGraph. It runs on AWS Lambda + API Gateway with no
always-on cost (pay only for what you use via Bedrock on-demand).

## Architecture

*Full diagram coming in M2/M4 — see [ARCHITECTURE.md](ARCHITECTURE.md).*

High-level data flow:
```
Documents → Ingestion → Chunking → Bedrock Embeddings → FAISS Index (S3)
                                                              ↓
User Query → API Gateway → Lambda → LangGraph Agent Graph → Response
                                    ├── Router Agent
                                    ├── Retriever Agent (FAISS)
                                    ├── Tool Agent (external APIs)
                                    └── Critic Agent (groundedness check)
```

## Local Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- AWS CLI configured (for M4+ deployment)

### Install

```bash
git clone https://github.com/BhanuSandeepVellalacheruvu/AgentRAG.git
cd AgentRAG
cp .env.example .env   # fill in your real values
uv sync --extra dev
```

### Run Tests

```bash
uv run pytest
```

### Lint & Format

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/ tests/
```

### Run the API Locally (M3+)

```bash
uv run uvicorn agentrag.api.main:app --reload
```

## Deploying (M4+)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full deploy/teardown instructions.

**Always tear down between demo sessions to avoid idle charges:**

```bash
cd src/agentrag/infra
uv run cdk destroy --all
```

## Cost Profile

All resources are **zero-cost by default** (AWS free tier or pure on-demand):
- Lambda: 1M requests/month free
- API Gateway HTTP API: 1M requests/month free
- S3: 5 GB storage + 20k GET + 2k PUT free
- DynamoDB: 25 GB + 200M requests free
- Bedrock: on-demand only — the only billable line item at actual usage
- A **$2/month Budget alarm** fires before anything significant accrues

## License

MIT — see [LICENSE](LICENSE).
