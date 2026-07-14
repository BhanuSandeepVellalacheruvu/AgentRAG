# AgentRAG

> A production-grade multi-agent Retrieval-Augmented Generation (RAG) system
> orchestrated with LangGraph and deployed on zero-cost AWS serverless infrastructure.

## What Is This?

AgentRAG is a multi-agent AI pipeline that answers complex HR questions by intelligently
routing queries through retrieval, tool-calling, and critique agents — all
orchestrated with LangGraph. It runs on AWS Lambda + API Gateway with no
always-on cost (pay only for actual usage via Bedrock on-demand).

## Architecture & Data Flow

*For detailed flow diagrams, sequencing, and topology, see [ARCHITECTURE.md](ARCHITECTURE.md).*

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
- Node.js & npm (for CDK deployment)

### Install

```bash
git clone https://github.com/BhanuSandeepVellalacheruvu/AgentRAG.git
cd AgentRAG
cp .env.example .env   # Fill in your AWS credentials and keys
uv sync --extra dev
```

### Run Tests & Linting

```bash
# Run pytest with code coverage (enforcing >=80% baseline)
PYTHONPATH=src uv run pytest

# Check code format and styling
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/ tests/
```

### Run the API & static UI locally

```bash
# Start hot-reloading development server
PYTHONPATH=src uv run uvicorn agentrag.api.main:app --reload
```
Open `http://localhost:8000` in your browser to interact with the responsive, glassmorphic UI, search documents, and view real-time LangGraph audit logs.

### Run Evaluation Harness

To run the automated RAG evaluation harness and generate latency/groundedness report (`eval_report.json`):
```bash
PYTHONPATH=src uv run python -m agentrag.observability.evaluation
```

---

## Deploying to AWS (CDK)

Infrastructure is defined in [src/agentrag/infra/](src/agentrag/infra).

```bash
# CD into infrastructure directory
cd src/agentrag/infra

# Deploy the stack (builds Docker container asset and provisions S3, DynamoDB, Lambda, and API GW)
npx aws-cdk deploy
```

**Always tear down between demo sessions to avoid any cost accrual:**
```bash
# Destroy all provisioned stack resources
npx aws-cdk destroy
```

---

## Cost Profile

All resources are **zero-cost by default** when idle, leveraging standard free tier limits and pure pay-per-request pricing:
- **AWS Lambda**: 1 million free requests per month, 400k GB-sec
- **API Gateway HTTP API**: 1 million requests per month free
- **S3 Standard Storage**: 5 GB standard storage + 20k GET + 2k PUT free
- **DynamoDB Tables**: 25 GB storage + 25 WCU + 25 RCU free
- **AWS Budget**: An automated **$0.50/month Budget alarm** notifies you immediately via email if actual or forecasted AWS costs exceed $0.50.
- **Amazon Bedrock**: pay-per-token on-demand (Claude 3 Haiku and Titan Embeddings).

## License

MIT — see [LICENSE](LICENSE).
