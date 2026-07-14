"""Pydantic schemas for the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the UI."""

    query: str = Field(..., description="The user's query.")
    session_id: str | None = Field(None, description="Optional session ID for tracing.")


class ChatResponse(BaseModel):
    """Response returned to the UI."""

    reply: str = Field(..., description="The AI's response text.")
    steps: list[str] = Field(
        ..., description="The sequence of LangGraph nodes executed."
    )
    grounded: bool = Field(
        ..., description="Whether the response passed the critic's hallucination check."
    )
