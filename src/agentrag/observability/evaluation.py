"""Evaluation harness for AgentRAG."""

import argparse
import json
import time
from typing import Any

from agentrag.agents.graph import graph

# Sample Q&A dataset derived from the HR policy dataset
EVAL_DATASET = [
    {
        "query": "What happens if the laws change before the annual policy review?",
        "reference_answer": "The company reviews its policies annually, but if laws change, the policies may be updated sooner to maintain compliance.",  # noqa: E501
    },
    {
        "query": "How often are overtime trends reviewed by management and HR?",
        "reference_answer": "Management and HR review overtime trends regularly to adjust staffing and prevent employee burnout.",  # noqa: E501
    },
    {
        "query": "Can you explain the approval process for compensatory off?",
        "reference_answer": "To apply for comp off, you must send an email to your manager with the date/hours of overtime worked and the project name.",  # noqa: E501
    },
    {
        "query": "Who can I contact at Kreeda Labs if I want to report bribery?",
        "reference_answer": "You should contact the Compliance Officer or report it through the company's whistleblower hotline.",  # noqa: E501
    },
    {
        "query": "What is the penalty if a contractor violates the bribery policy?",
        "reference_answer": "The company may take disciplinary action or terminate the contractor's engagement immediately.",  # noqa: E501
    },
]


def run_evaluation() -> dict[str, Any]:
    """Execute evaluation on the test dataset and return metrics."""
    results = []
    total_latency = 0.0
    grounded_count = 0

    print(f"Starting evaluation on {len(EVAL_DATASET)} test cases...")

    for i, test_case in enumerate(EVAL_DATASET, 1):
        query = test_case["query"]
        ref_answer = test_case["reference_answer"]

        print(f"\n[{i}/{len(EVAL_DATASET)}] Query: {query}")

        start_time = time.perf_counter()
        # Invoke the LangGraph pipeline
        state = {"query": query}
        final_state = graph.invoke(state)
        latency = time.perf_counter() - start_time
        total_latency += latency

        generation = final_state.get("generation", "")
        grounded = final_state.get("grounded", False)
        steps = final_state.get("steps", [])

        if grounded:
            grounded_count += 1

        print(f"  Grounded: {grounded}")
        print(f"  Latency: {latency:.3f}s")
        print(f"  Reply: {generation[:100]}...")

        results.append(
            {
                "query": query,
                "reference_answer": ref_answer,
                "generated_answer": generation,
                "grounded": grounded,
                "latency_s": latency,
                "steps": steps,
            }
        )

    avg_latency = total_latency / len(EVAL_DATASET)
    groundedness_rate = grounded_count / len(EVAL_DATASET)

    report = {
        "dataset_size": len(EVAL_DATASET),
        "average_latency_s": avg_latency,
        "groundedness_rate": groundedness_rate,
        "results": results,
    }

    return report


def main() -> None:
    """CLI entrypoint for running evaluation."""
    parser = argparse.ArgumentParser(description="AgentRAG Evaluation Harness")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.json",
        help="Path to save the JSON evaluation report.",
    )
    args = parser.parse_args()

    report = run_evaluation()

    print("\n" + "=" * 40)
    print("Evaluation Complete!")
    print(f"Average Latency: {report['average_latency_s']:.3f}s")
    print(f"Groundedness Rate: {report['groundedness_rate'] * 100.0:.1f}%")
    print("=" * 40)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
