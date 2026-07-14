"""Tests for the API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from agentrag.api.main import app

client = TestClient(app)


def test_static_files() -> None:
    """Test that index.html is served from root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AgentRAG" in response.text


@patch("agentrag.api.routes.graph")
def test_chat_endpoint_success(mock_graph: MagicMock) -> None:
    """Test successful chat response from API."""
    mock_graph.invoke.return_value = {
        "generation": "Test answer from AI.",
        "steps": ["routed_to_direct_answer", "generated_response", "bypassed_critique"],
        "grounded": True,
    }

    payload = {"query": "hello"}
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Test answer from AI."
    assert data["steps"] == [
        "routed_to_direct_answer",
        "generated_response",
        "bypassed_critique",
    ]
    assert data["grounded"] is True
    mock_graph.invoke.assert_called_once_with({"query": "hello"})


@patch("agentrag.api.routes.graph")
def test_chat_endpoint_failure(mock_graph: MagicMock) -> None:
    """Test error handling in chat endpoint."""
    mock_graph.invoke.side_effect = Exception("Bedrock error")

    payload = {"query": "hello"}
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 500
    data = response.json()
    assert "Bedrock error" in data["detail"]
