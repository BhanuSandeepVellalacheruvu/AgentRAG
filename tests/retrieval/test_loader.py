"""Tests for src/agentrag/retrieval/loader.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentrag.retrieval.loader import DocumentLoader


@pytest.fixture
def temp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with test documents."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Valid JSONL
    jsonl_content = [
        {"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "Hi there!"}]},
        {"messages": [{"role": "user", "content": "What is the policy?"}, {"role": "assistant", "content": "The policy is X."}]}
    ]
    with open(docs_dir / "data.jsonl", "w") as f:
        f.write("\n") # Empty line
        for item in jsonl_content:
            f.write(json.dumps(item) + "\n")
            
    # Invalid JSONL missing user or assistant
    with open(docs_dir / "bad.jsonl", "w") as f:
        f.write("not valid json\n")
        f.write(json.dumps({"messages": []}) + "\n") # Empty messages
        f.write(json.dumps({"messages": [{"role": "system", "content": "x"}]}) + "\n") # Missing user/assistant
        f.write(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n") # Missing assistant
        
    # Valid Markdown
    with open(docs_dir / "doc.md", "w") as f:
        f.write("# Heading\n\nSome text here.")
        
    # Empty Text
    with open(docs_dir / "empty.txt", "w") as f:
        f.write("   \n")
        
    # Bad encoding text
    with open(docs_dir / "bad_encoding.txt", "wb") as f:
        f.write(b"\xff\xfe\x00\x00")

    # Unrecognized extension
    with open(docs_dir / "ignore.csv", "w") as f:
        f.write("a,b,c\n")
        
    return docs_dir


def test_load_local_directory(temp_docs_dir: Path) -> None:
    """Test loading documents from a local directory."""
    loader = DocumentLoader()
    docs = list(loader.load_local_directory(temp_docs_dir))
    
    # Expect 2 from data.jsonl, 1 from doc.md
    assert len(docs) == 3
    
    jsonl_doc1 = next(d for d in docs if d.filename == "data.jsonl" and "Hi there!" in d.text)
    assert jsonl_doc1.text == "Q: Hello?\nA: Hi there!"
    assert jsonl_doc1.source == "data.jsonl:2"
    
    jsonl_doc2 = next(d for d in docs if d.filename == "data.jsonl" and "policy is X" in d.text)
    assert jsonl_doc2.text == "Q: What is the policy?\nA: The policy is X."
    
    md_doc = next(d for d in docs if d.filename == "doc.md")
    assert md_doc.text == "# Heading\n\nSome text here."
    assert md_doc.source == str(temp_docs_dir / "doc.md")


def test_load_local_directory_not_found() -> None:
    """Test loading from a non-existent directory raises FileNotFoundError."""
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        list(loader.load_local_directory("/does/not/exist/ever"))


@patch("boto3.client")
def test_load_s3_prefix(mock_boto3_client: MagicMock) -> None:
    """Test loading documents from S3."""
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3
    
    # Mock pagination
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "docs/data.jsonl"},
                {"Key": "docs/"}, # Directory object, should be skipped
                {"Key": "docs/readme.txt"},
                {"Key": "docs/empty.txt"},
            ]
        },
        {
            # Page without Contents to test edge case
        }
    ]
    
    # Mock get_object responses
    def mock_get_object(Bucket, Key):
        if Key == "docs/data.jsonl":
            content = "\n" + json.dumps({"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]})
            return {"Body": MagicMock(read=MagicMock(return_value=content.encode("utf-8")))}
        elif Key == "docs/readme.txt":
            return {"Body": MagicMock(read=MagicMock(return_value=b"Hello text"))}
        elif Key == "docs/empty.txt":
            return {"Body": MagicMock(read=MagicMock(return_value=b"  \n"))}
        return {"Body": MagicMock(read=MagicMock(return_value=b""))}
        
    mock_s3.get_object.side_effect = mock_get_object
    
    loader = DocumentLoader(s3_bucket="test-bucket", aws_region="us-east-1")
    docs = list(loader.load_s3_prefix("docs/"))
    
    assert len(docs) == 2
    assert docs[0].filename == "data.jsonl"
    assert docs[0].text == "Q: q1\nA: a1"
    assert docs[0].source == "data.jsonl:2"
    
    assert docs[1].filename == "readme.txt"
    assert docs[1].text == "Hello text"
    assert docs[1].source == "s3://test-bucket/docs/readme.txt"


def test_load_s3_prefix_no_bucket() -> None:
    """Test loading from S3 without bucket configured raises ValueError."""
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="S3 bucket not configured"):
        list(loader.load_s3_prefix("docs/"))
