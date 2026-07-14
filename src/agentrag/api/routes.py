"""API routes for AgentRAG."""

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

        # Invoke the LangGraph synchronously for now (Lambda is typically sync or wrapped)  # noqa: E501
        final_state = graph.invoke(initial_state)

        return ChatResponse(
            reply=final_state.get("generation", "Error: No response generated."),
            steps=final_state.get("steps", []),
            grounded=final_state.get("grounded", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
