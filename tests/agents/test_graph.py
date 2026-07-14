"""Tests for src/agentrag/agents/graph.py."""

from langgraph.graph import END

from agentrag.agents.graph import check_groundedness, graph, should_retrieve


def test_should_retrieve() -> None:
    """Test should_retrieve conditional edge."""
    state1 = {"next_action": "retrieve"}
    assert should_retrieve(state1) == "retrieve"  # type: ignore

    state2 = {"next_action": "direct_answer"}
    assert should_retrieve(state2) == "direct_answer"  # type: ignore

    # Tool falls back to direct_answer for now
    state3 = {"next_action": "tool"}
    assert should_retrieve(state3) == "direct_answer"  # type: ignore


def test_check_groundedness() -> None:
    """Test check_groundedness conditional edge."""
    # Grounded ends
    state1 = {"grounded": True, "steps": ["generated_response"]}
    assert check_groundedness(state1) == END  # type: ignore

    # Ungrounded generates again
    state2 = {"grounded": False, "steps": ["generated_response"]}
    assert check_groundedness(state2) == "generate"  # type: ignore

    # After 2 generations, force END to prevent infinite loop
    state3 = {"grounded": False, "steps": ["generated_response", "generated_response"]}
    assert check_groundedness(state3) == END  # type: ignore


def test_graph_compiled() -> None:
    """Test the graph is properly compiled."""
    assert graph is not None
    # We can check the nodes in the graph
    # LangGraph StateGraph compiled into a graph object
    assert "router" in graph.nodes
    assert "retriever" in graph.nodes
    assert "generator" in graph.nodes
    assert "critic" in graph.nodes
