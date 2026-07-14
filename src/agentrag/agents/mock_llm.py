"""Mock Bedrock LLM for local offline development."""

from typing import Any


class MockResponse:
    """Mock LLM response mimicking LangChain's AIMessage."""

    def __init__(self, content: str) -> None:
        """Initialize mock response."""
        self.content = content


class MockChatBedrock:
    """Mock ChatBedrock class that mimics Bedrock invokes without calling AWS."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize mock model."""
        self.model_id = model_id

    def invoke(self, messages: Any) -> MockResponse:
        """Mock invocation returning static answer or custom query responses."""
        # Simple rule-based mock response based on user input
        user_query = ""
        for msg in messages:
            if isinstance(msg, tuple) and msg[0] == "human":
                user_query = msg[1].lower()
            elif hasattr(msg, "content") and not getattr(msg, "type", "") == "system":
                user_query = str(msg.content).lower()

        if "policy" in user_query or "vacation" in user_query:
            content = "According to the HR policy, full-time employees receive 20 days of paid vacation per year."
        else:
            content = "This is a mock response from the offline AgentRAG model. How can I assist you with HR policies?"

        return MockResponse(content)

    def with_structured_output(self, schema: type) -> Any:
        """Mock with_structured_output method."""
        class MockStructuredRunnable:
            def invoke(self, messages: Any) -> Any:
                schema_name = schema.__name__
                
                # Fetch human query to make router decisions dynamic
                user_query = ""
                for msg in messages:
                    if isinstance(msg, tuple) and msg[0] == "human":
                        user_query = msg[1].lower()
                    elif hasattr(msg, "content") and not getattr(msg, "type", "") == "system":
                        user_query = str(msg.content).lower()

                if schema_name == "RouteDecision":
                    # Route to retrieve if they ask about policy/vacation
                    if "policy" in user_query or "hr" in user_query or "vacation" in user_query:
                        return schema(next_action="retrieve")
                    return schema(next_action="direct_answer")
                elif schema_name == "Critique":
                    return schema(is_grounded=True)
                return schema()

        return MockStructuredRunnable()
