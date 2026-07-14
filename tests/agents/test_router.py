"""Tests for src/agentrag/agents/router.py."""

from unittest.mock import MagicMock, patch

from agentrag.agents.router import RouteDecision, route_query


@patch("agentrag.agents.router.ChatBedrock")
@patch("agentrag.agents.router.get_settings")
def test_route_query_success(
    mock_get_settings: MagicMock, mock_chat_bedrock: MagicMock
) -> None:
    """Test successful routing."""
    mock_settings = MagicMock()
    mock_settings.aws_region = "us-east-1"
    mock_settings.use_mock_llm = False
    mock_get_settings.return_value = mock_settings

    mock_llm = MagicMock()
    mock_chat_bedrock.return_value = mock_llm

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Mock LLM return value
    mock_structured.invoke.return_value = RouteDecision(next_action="retrieve")

    state = {"query": "What is the leave policy?"}
    new_state = route_query(state)  # type: ignore

    assert new_state["next_action"] == "retrieve"
    assert new_state["steps"] == ["routed_to_retrieve"]
    assert new_state["query"] == "What is the leave policy?"
    mock_structured.invoke.assert_called_once()


@patch("agentrag.agents.router.ChatBedrock")
@patch("agentrag.agents.router.get_settings")
def test_route_query_fallback_on_error(
    mock_get_settings: MagicMock, mock_chat_bedrock: MagicMock
) -> None:
    """Test router falls back to 'retrieve' on LLM error."""
    mock_settings = MagicMock()
    mock_settings.aws_region = "us-east-1"
    mock_settings.use_mock_llm = False
    mock_get_settings.return_value = mock_settings

    mock_llm = MagicMock()
    mock_chat_bedrock.return_value = mock_llm

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Simulate an error
    mock_structured.invoke.side_effect = Exception("API error")

    state = {"query": "Unknown query"}
    new_state = route_query(state)  # type: ignore

    assert new_state["next_action"] == "retrieve"
    assert new_state["steps"] == ["routed_to_retrieve"]
