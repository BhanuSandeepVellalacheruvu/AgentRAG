"""LangGraph setup for AgentRAG."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentrag.agents.critic import critique_generation
from agentrag.agents.generator import generate_response
from agentrag.agents.retriever import retrieve_context
from agentrag.agents.router import route_query
from agentrag.agents.state import AgentState


def should_retrieve(state: AgentState) -> str:
    """Return the next action from the state."""
    # Ensure tool falls back to generator for now until tools are implemented
    action = state.get("next_action", "retrieve")
    if action == "tool":
        return "direct_answer"
    return action


def check_groundedness(state: AgentState) -> str:
    """Return END if grounded, else loop back to generate."""
    # We prevent infinite loops by checking steps
    # If we already tried generating twice, just end
    generations = sum(
        1 for step in state.get("steps", []) if step == "generated_response"
    )
    if state.get("grounded", True) or generations >= 2:
        return END
    return "generate"


def build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", route_query)
    workflow.add_node("retriever", retrieve_context)
    workflow.add_node("generator", generate_response)
    workflow.add_node("critic", critique_generation)

    # Entry point
    workflow.set_entry_point("router")

    # Routing edges
    workflow.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retriever",
            "direct_answer": "generator",
        },
    )

    # Sequential edges
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "critic")

    # Loop back or end
    workflow.add_conditional_edges(
        "critic", check_groundedness, {END: END, "generate": "generator"}
    )

    return workflow.compile()


# Create the compiled graph instance
graph = build_graph()
