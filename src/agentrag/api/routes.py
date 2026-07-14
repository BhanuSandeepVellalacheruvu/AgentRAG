"""API routes for AgentRAG."""

from typing import Any

from fastapi import APIRouter, HTTPException

from agentrag.agents.graph import graph
from agentrag.api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle incoming chat requests by routing them through the LangGraph."""
    try:
        # Initialize state with the query
        initial_state = {"query": request.query}

        # Invoke the LangGraph synchronously for now (Lambda is typically sync or wrapped)
        final_state = graph.invoke(initial_state)

        return ChatResponse(
            reply=final_state.get("generation", "Error: No response generated."),
            steps=final_state.get("steps", []),
            grounded=final_state.get("grounded", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def status_endpoint() -> dict[str, Any]:
    """Return status and configuration details of the backend."""
    from agentrag.config import get_settings

    settings = get_settings()
    return {
        "use_mock_llm": settings.use_mock_llm,
        "use_mock_embeddings": settings.use_mock_embeddings,
        "aws_region": settings.aws_region,
    }
