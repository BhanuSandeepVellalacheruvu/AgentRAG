#!/bin/bash
set -e

# Initial commit on main
echo "# AgentRAG" > README.md
git add README.md
git commit -m "Initial commit" || true

# M0 – Project Scaffolding
git checkout -b feature/m0-project-scaffolding
mkdir -p .github/workflows src tests
touch pyproject.toml Dockerfile .ruff.toml mypy.ini .pre-commit-config.yaml .github/workflows/ci.yml
git add .
git commit -m "M0: Project Scaffolding"
git checkout main

# M1 – Ingestion & Retrieval
git checkout -b feature/m1-ingestion-retrieval
mkdir -p src/ingestion src/retrieval
touch src/ingestion/parser.py src/ingestion/chunking.py src/retrieval/embeddings.py src/retrieval/faiss_store.py tests/test_search.py
git add .
git commit -m "M1: Ingestion & Retrieval"
git checkout main

# M2 – Agent Orchestration
git checkout -b feature/m2-agent-orchestration
mkdir -p src/agents
touch src/agents/router_agent.py src/agents/retriever_agent.py src/agents/tool_agent.py src/agents/critic_agent.py src/agents/workflow.py
git add .
git commit -m "M2: Agent Orchestration"
git checkout main

# M3 – API & UI
git checkout -b feature/m3-api-ui
mkdir -p src/api src/ui
touch src/api/main.py src/api/auth.py src/api/endpoints.py src/ui/index.html src/ui/chat.js
git add .
git commit -m "M3: API & UI"
git checkout main

# M4 – AWS Infrastructure
git checkout -b feature/m4-aws-infrastructure
mkdir -p infrastructure
touch infrastructure/app.py infrastructure/cdk_stack.py infrastructure/lambda_handler.py infrastructure/budget.json
git add .
git commit -m "M4: AWS Infrastructure"
git checkout main

# M5 – CI/CD
git checkout -b feature/m5-cicd
mkdir -p .github/workflows
touch .github/workflows/deploy.yml .github/workflows/test.yml .github/workflows/release.yml
git add .
git commit -m "M5: CI/CD"
git checkout main

# M6 – Observability
git checkout -b feature/m6-observability-evaluation
mkdir -p src/observability
touch src/observability/cloudwatch.py src/observability/langsmith_client.py src/observability/logger.py src/observability/tracing.py src/observability/eval.py
git add .
git commit -m "M6: Observability"
git checkout main

# M7 – Documentation
git checkout -b feature/m7-documentation-polish
touch ARCHITECTURE.md DEMO.md RESUME_BULLETS.md COST_ANALYSIS.md
git add .
git commit -m "M7: Documentation"
git checkout main

