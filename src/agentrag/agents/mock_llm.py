"""Mock Bedrock LLM for local offline development."""

import re
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
        system_content = ""
        user_query = ""
        for msg in messages:
            if isinstance(msg, tuple):
                if msg[0] == "system":
                    system_content = msg[1]
                elif msg[0] == "human":
                    user_query = msg[1]
            elif hasattr(msg, "content"):
                if getattr(msg, "type", "") == "system":
                    system_content = str(msg.content)
                else:
                    user_query = str(msg.content)

        # 1. Smart Retrieval-based mock answering:
        # If context was retrieved from the local FAISS/BM25 store,
        # extract the exact answer from the top chunk!
        if "context:" in system_content.lower():
            # Extract the first answer block 'A: ...' from the context
            match = re.search(
                r"A:\s*(.*?)(?=\n\n|\nSource:|$)", system_content, re.DOTALL
            )
            if match:
                answer = match.group(1).strip()
                if answer:
                    return MockResponse(answer)

        # 2. Rule-based fallbacks if no context was retrieved
        user_query_lower = user_query.lower()
        if "compensatory" in user_query_lower or "comp off" in user_query_lower:
            content = (
                "When applying for compensatory off (comp off), "
                "your email to your manager should include:\n"
                "1. The date(s) and hours of the extra work/overtime performed.\n"
                "2. The specific project name or reason for the overtime.\n"
                "3. The proposed date(s) you wish to take as compensatory off.\n"
                "4. A reference to the prior approval received for the overtime."
            )
        elif (
            "unethical" in user_query_lower
            or "ethics" in user_query_lower
            or "conduct" in user_query_lower
        ):
            content = (
                "For ethical concerns or questions about unethical behavior, "
                "you should consult the company's Code of Conduct and Ethics Policy, "
                "or reach out to the compliance department. Keeping workplace "
                "integrity high is our priority, and all reports can be submitted "
                "confidentially."
            )
        elif (
            "review" in user_query_lower
            or "annual" in user_query_lower
            or "why" in user_query_lower
        ):
            content = (
                "The company reviews its policies annually to ensure alignment with "
                "local labor regulations, incorporate feedback from the teams, and "
                "maintain competitive standards for the organization."
            )
        elif "vacation" in user_query_lower or "annual leave" in user_query_lower:
            content = (
                "According to the HR policy, full-time employees receive "
                "20 days of paid vacation per year."
            )
        elif "sick" in user_query_lower or "medical" in user_query_lower:
            content = (
                "Sick leave covers up to 10 paid days per year for medical recovery. "
                "Absences over 3 consecutive days require a doctor's note."
            )
        elif "leave" in user_query_lower or "policy" in user_query_lower:
            content = (
                "All standard company policies (including leave, code of conduct, "
                "and remote work) are documented in the Employee Handbook. "
                "Let me know if you want information on a specific policy!"
            )
        else:
            content = (
                "This is a mock response from the offline AgentRAG model. "
                "How can I assist you with HR policies?"
            )

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
                    elif (
                        hasattr(msg, "content")
                        and not getattr(msg, "type", "") == "system"
                    ):
                        user_query = str(msg.content).lower()

                if schema_name == "RouteDecision":
                    # Route to retrieve if they ask about policy, vacation, etc.
                    keywords = [
                        "policy",
                        "hr",
                        "vacation",
                        "leave",
                        "sick",
                        "medical",
                        "compensatory",
                        "comp off",
                        "off",
                        "ethics",
                        "conduct",
                        "bribe",
                        "bribery",
                        "kreeda",
                        "labs",
                        "report",
                        "contact",
                        "whistleblower",
                        "violation",
                        "contractor",
                    ]
                    if any(kw in user_query for kw in keywords):
                        return schema(next_action="retrieve")
                    return schema(next_action="direct_answer")
                elif schema_name == "Critique":
                    return schema(is_grounded=True)
                return schema()

        return MockStructuredRunnable()
