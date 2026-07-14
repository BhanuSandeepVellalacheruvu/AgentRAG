"""Critic agent for the LangGraph pipeline."""

from typing import Any, cast

from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field

from agentrag.agents.mock_llm import MockChatBedrock
from agentrag.agents.state import AgentState
from agentrag.config import get_settings


class Critique(BaseModel):
    """Critique decision on groundedness."""

    is_grounded: bool = Field(
        ...,
        description="True if the generation is fully grounded in the provided context, "
        "or if the generation correctly states that the answer cannot be found in the context. "  # noqa: E501
        "False if the generation hallucinates information not present in the context.",
    )


def critique_generation(state: AgentState) -> AgentState:
    """Verify if the generation is grounded in the retrieved context."""
    # If there was no context (e.g. direct answer), we bypass critique
    if not state.get("context"):
        return {**state, "grounded": True, "steps": ["bypassed_critique"]}

    settings = get_settings()

    # We use a fast/cheap model for critique
    if settings.use_mock_llm:
        llm: Any = MockChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    else:
        llm = ChatBedrock(  # type: ignore[call-arg]
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name=settings.aws_region,
            client=None,
        )

    critic_llm = llm.with_structured_output(Critique)

    context_str = "\n\n".join(
        f"Source: {c.chunk.source}\n{c.chunk.text}" for c in state["context"]
    )

    system_prompt = (
        "You are an expert fact-checker. You will be provided with a source context and an AI-generated answer. "  # noqa: E501
        "Determine if the AI-generated answer is fully supported by the source context."
    )

    user_prompt = f"Context:\n{context_str}\n\nAnswer:\n{state['generation']}"

    messages = [("system", system_prompt), ("human", user_prompt)]

    try:
        decision = cast(Critique, critic_llm.invoke(messages))
        is_grounded = decision.is_grounded
    except Exception:
        # If the critic fails, we tentatively accept it to prevent blocking
        is_grounded = True

    return {**state, "grounded": is_grounded, "steps": ["critiqued_generation"]}
