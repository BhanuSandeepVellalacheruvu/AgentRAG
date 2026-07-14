"""Tests for src/agentrag/retrieval/chunker.py."""

import pytest

from agentrag.retrieval.chunker import Chunker
from agentrag.retrieval.loader import Document


def test_chunker_init_validation() -> None:
    """Test chunker initialization validates parameters."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        Chunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        Chunker(chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        Chunker(chunk_size=100, chunk_overlap=100)


def test_chunk_empty_document() -> None:
    """Test chunking an empty document returns empty list."""
    chunker = Chunker()
    doc = Document(text="   ", filename="empty.txt", source="empty.txt")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunk_short_document() -> None:
    """Test chunking a document shorter than chunk size."""
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    doc = Document(text="Short text", filename="short.txt", source="short.txt:1")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text"
    assert chunks[0].chunk_index == 0


def test_chunk_document_with_overlap() -> None:
    """Test chunking creates overlapping chunks."""
    # Text is 30 characters long
    text = "012345678901234567890123456789"
    chunker = Chunker(chunk_size=15, chunk_overlap=5)
    doc = Document(text=text, filename="num.txt", source="num.txt:1")

    # 0-15: "012345678901234" (len 15)
    # Next start = 15 - 5 = 10
    # 10-25: "012345678901234" (starts at original index 10: "01234" from prev, + "5678901234") -> "012345678901234"  # noqa: E501
    # Wait, original text at 10 is "0123456789" (last 10 of first 20)
    # Let's check exactly:
    # 0123456789 0123456789 0123456789

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 3

    assert chunks[0].text == "012345678901234"
    assert chunks[0].chunk_index == 0

    assert chunks[1].text == "012345678901234" # starting at index 10: "012345678901234"
    assert chunks[1].chunk_index == 1

    assert chunks[2].text == "0123456789" # starting at index 20
    assert chunks[2].chunk_index == 2


def test_chunk_document_word_boundary() -> None:
    """Test chunking attempts to split on word boundaries."""
    # "This is a sentence. Another sentence here."
    # Let's say chunk size is 25.
    # 0..25 is "This is a sentence. Anoth"
    # The code should backtrack to the last space before 'Anoth', which is after '.'
    text = "This is a sentence. Another sentence here."
    chunker = Chunker(chunk_size=25, chunk_overlap=5)
    doc = Document(text=text, filename="words.txt", source="words.txt:1")

    chunks = chunker.chunk_document(doc)

    # Chunk 0 should backtrack to "This is a sentence."
    assert chunks[0].text == "This is a sentence."
    assert chunks[0].chunk_index == 0


def test_chunk_documents() -> None:
    """Test chunking multiple documents."""
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    docs = [
        Document(text="Doc one text", filename="1.txt", source="1.txt"),
        Document(text="Doc two text", filename="2.txt", source="2.txt")
    ]

    all_chunks = chunker.chunk_documents(docs)
    assert len(all_chunks) == 4 # Should have chunks from both
    assert all_chunks[0].filename == "1.txt"
    assert all_chunks[-1].filename == "2.txt"
