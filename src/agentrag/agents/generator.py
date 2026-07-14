"""Generator agent for the LangGraph pipeline."""

from typing import Any

from langchain_aws import ChatBedrock

from agentrag.agents.mock_llm import MockChatBedrock
from agentrag.agents.state import AgentState
from agentrag.config import get_settings


def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on context or directly from the query."""
    settings = get_settings()

    # We use Sonnet for generation because it provides better reasoning
    if settings.use_mock_llm:
        llm: Any = MockChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    else:
        llm = ChatBedrock(  # type: ignore[call-arg]
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=settings.aws_region,
            client=None,
        )

    query = state["query"]
    context = state.get("context", [])

    if context:
        context_str = "\n\n".join(
            f"Source: {c.chunk.source}\n{c.chunk.text}" for c in context
        )
        system_prompt = (
            "You are a helpful HR assistant. Answer the user's question based ONLY on the "  # noqa: E501
            "provided context. If the answer cannot be found in the context, say so.\n\n"  # noqa: E501
            f"Context:\n{context_str}"
        )
    else:
        system_prompt = (
            "You are a helpful HR assistant. Answer the user's question directly."
        )

    messages = [("system", system_prompt), ("human", query)]

    response = llm.invoke(messages)

    return {
        **state,
        "generation": str(response.content),
        "steps": ["generated_response"],
    }
