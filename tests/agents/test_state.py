"""Tests for src/agentrag/agents/state.py."""

from typing import get_type_hints

from agentrag.agents.state import AgentState


def test_agent_state_annotations() -> None:
    """Test AgentState has the expected keys and annotations."""
    hints = get_type_hints(AgentState, include_extras=True)

    assert "query" in hints
    assert "steps" in hints
    assert "context" in hints
    assert "generation" in hints
    assert "grounded" in hints
    assert "next_action" in hints
