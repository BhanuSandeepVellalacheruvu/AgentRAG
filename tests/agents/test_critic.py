"""Tests for src/agentrag/agents/critic.py."""

from unittest.mock import MagicMock, patch

from agentrag.agents.critic import Critique, critique_generation
from agentrag.retrieval.chunker import Chunk
from agentrag.retrieval.faiss_store import SearchResult


def test_critique_bypassed_no_context() -> None:
    """Test critique is bypassed if context is empty."""
    state = {"query": "hi", "generation": "hello", "context": []}
    new_state = critique_generation(state)  # type: ignore

    assert new_state["grounded"] is True
    assert new_state["steps"] == ["bypassed_critique"]


@patch("agentrag.agents.critic.ChatBedrock")
@patch("agentrag.agents.critic.get_settings")
def test_critique_generation_grounded(
    mock_get_settings: MagicMock, mock_chat_bedrock: MagicMock
) -> None:
    """Test critique returns True for grounded answer."""
    mock_get_settings.return_value = MagicMock(aws_region="us-east-1")

    mock_llm = MagicMock()
    mock_chat_bedrock.return_value = mock_llm

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_structured.invoke.return_value = Critique(is_grounded=True)

    state = {
        "query": "policy?",
        "generation": "It's X.",
        "context": [
            SearchResult(
                chunk=Chunk(
                    text="policy is X", filename="a.txt", source="a", chunk_index=0
                ),
                score=1.0,
            )
        ],
    }

    new_state = critique_generation(state)  # type: ignore

    assert new_state["grounded"] is True
    assert new_state["steps"] == ["critiqued_generation"]


@patch("agentrag.agents.critic.ChatBedrock")
@patch("agentrag.agents.critic.get_settings")
def test_critique_fallback_on_error(
    mock_get_settings: MagicMock, mock_chat_bedrock: MagicMock
) -> None:
    """Test critique falls back to True on LLM error."""
    mock_get_settings.return_value = MagicMock(aws_region="us-east-1")

    mock_llm = MagicMock()
    mock_chat_bedrock.return_value = mock_llm

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_structured.invoke.side_effect = Exception("API Error")

    state = {
        "query": "policy?",
        "generation": "It's X.",
        "context": [
            SearchResult(
                chunk=Chunk(
                    text="policy is X", filename="a.txt", source="a", chunk_index=0
                ),
                score=1.0,
            )
        ],
    }

    new_state = critique_generation(state)  # type: ignore

    assert new_state["grounded"] is True
    assert new_state["steps"] == ["critiqued_generation"]
