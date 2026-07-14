"""Router agent for the LangGraph pipeline."""

from typing import Literal

from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field

from agentrag.agents.state import AgentState
from agentrag.config import get_settings


class RouteDecision(BaseModel):
    """Decision for the next routing step."""

    next_action: Literal["retrieve", "direct_answer", "tool"] = Field(
        ...,
        description="The next action to take based on the user query. "
        "Choose 'retrieve' for internal knowledge (HR policies, company info), "
        "'tool' for dynamic external data, or "
        "'direct_answer' for general conversational questions.",
    )


def route_query(state: AgentState) -> AgentState:
    """Determine the next step based on the query."""
    settings = get_settings()

    # Initialize the Bedrock model
    # We use a fast/cheap model like Haiku for routing
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name=settings.aws_region,
        client=None,  # will use default boto3 session
    )

    # Bind structured output
    router_llm = llm.with_structured_output(RouteDecision)

    system_prompt = (
        "You are an expert routing assistant. Analyze the user query and decide the next step.\n"  # noqa: E501
        "Options:\n"
        "1. retrieve: If the query asks about HR policies, internal guidelines, or company-specific knowledge.\n"  # noqa: E501
        "2. tool: If the query requires looking up external real-time information (e.g., weather, stocks).\n"  # noqa: E501
        "3. direct_answer: If the query is a greeting, casual chat, or general knowledge that needs no context."  # noqa: E501
    )

    messages = [("system", system_prompt), ("human", state["query"])]

    try:
        decision = router_llm.invoke(messages)
        next_action = decision.next_action
    except Exception:
        # Fallback to retrieve on error
        next_action = "retrieve"

    return {**state, "next_action": next_action, "steps": ["routed_to_" + next_action]}
