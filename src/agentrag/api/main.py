"""FastAPI application for AgentRAG."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mangum import Mangum

from agentrag.api.routes import router

app = FastAPI(
    title="AgentRAG API",
    description="Multi-agent RAG application deployed on AWS Bedrock and Lambda.",
    version="0.1.0",
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static UI
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)  # Ensure it exists

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# Mangum adapter for AWS Lambda
handler = Mangum(app)
