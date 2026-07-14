"""Tests for src/agentrag/agents/generator.py."""

from unittest.mock import MagicMock, patch

from agentrag.agents.generator import generate_response
from agentrag.retrieval.chunker import Chunk
from agentrag.retrieval.faiss_store import SearchResult


@patch("agentrag.agents.generator.ChatBedrock")
@patch("agentrag.agents.generator.get_settings")
def test_generate_response_with_context(
    mock_get_settings: MagicMock, mock_chat_bedrock: MagicMock
) -> None:
    """Test generating a response with context."""
    mock_settings = MagicMock()
    mock_settings.aws_region = "us-east-1"
    mock_get_settings.return_value = mock_settings

    mock_llm = MagicMock()
    mock_chat_bedrock.return_value = mock_llm

    mock_response = MagicMock()
    mock_response.content = "Here is the policy info."
    mock_llm.invoke.return_value = mock_response

    state = {
        "query": "policy?",
        "context": [
            SearchResult(
                chunk=Chunk(
                    text="policy info", filename="a.txt", source="a", chunk_index=0
                ),
                score=1.0,
            )
        ],
    }

    new_state = generate_response(state)  # type: ignore

    assert new_state["generation"] == "Here is the policy info."
    assert new_state["steps"] == ["generated_response"]

    # Check that LLM was called with the context
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2
    assert "policy info" in call_args[0][1]


@patch("agentrag.agents.generator.ChatBedrock")
@patch("agentrag.agents.generator.get_settings")
def test_generate_response_no_context(
    mock_get_settings: MagicMock, mock_chat_bedrock: MagicMock
) -> None:
    """Test generating a response directly without context."""
    mock_get_settings.return_value = MagicMock(aws_region="us-east-1")

    mock_llm = MagicMock()
    mock_chat_bedrock.return_value = mock_llm

    mock_response = MagicMock()
    mock_response.content = "Hello there!"
    mock_llm.invoke.return_value = mock_response

    state = {"query": "hi", "context": []}

    new_state = generate_response(state)  # type: ignore

    assert new_state["generation"] == "Hello there!"
    assert new_state["steps"] == ["generated_response"]

    # Check that LLM was called without context
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2
    assert "Context:" not in call_args[0][1]
