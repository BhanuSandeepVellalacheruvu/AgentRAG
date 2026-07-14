"""State definition for the AgentRAG LangGraph."""

import operator
from typing import Annotated, TypedDict

from agentrag.retrieval.faiss_store import SearchResult


class AgentState(TypedDict):
    """The state of the agent graph execution."""

    query: str
    # Use Annotated with operator.add to append steps instead of overwriting
    steps: Annotated[list[str], operator.add]
    context: list[SearchResult]
    generation: str
    grounded: bool
    # Internal routing flag
    next_action: str
