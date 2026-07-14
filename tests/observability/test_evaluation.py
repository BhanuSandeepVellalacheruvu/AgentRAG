"""Tests for the evaluation harness."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

from agentrag.observability.evaluation import main, run_evaluation


@patch("agentrag.observability.evaluation.graph.invoke")
def test_run_evaluation(mock_invoke: MagicMock) -> None:
    """run_evaluation calls invoke and returns report."""
    mock_invoke.return_value = {
        "generation": "mock answer",
        "grounded": True,
        "steps": ["step1"],
    }

    report = run_evaluation()

    assert report["dataset_size"] == 5
    assert report["groundedness_rate"] == 1.0
    assert len(report["results"]) == 5
    assert report["results"][0]["generated_answer"] == "mock answer"
    assert report["results"][0]["grounded"] is True


@patch("agentrag.observability.evaluation.run_evaluation")
def test_main_cli(mock_run: MagicMock, tmp_path: Any) -> None:
    """main CLI writes json report to file."""
    mock_run.return_value = {
        "average_latency_s": 0.5,
        "groundedness_rate": 0.8,
        "results": [],
    }
    output_file = tmp_path / "report.json"

    with patch("sys.argv", ["evaluation.py", "--output", str(output_file)]):
        main()

    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)
    assert data == mock_run.return_value
