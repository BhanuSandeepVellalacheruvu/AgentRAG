"""Document chunking module.

Splits loaded documents into smaller overlapping chunks for embedding and retrieval.
"""

from dataclasses import dataclass

from agentrag.retrieval.loader import Document


@dataclass
class Chunk:
    """A segment of text from a Document, ready for embedding."""

    text: str
    filename: str
    source: str
    chunk_index: int


class Chunker:
    """Splits documents into overlapping character chunks."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single Document into multiple Chunks."""
        text = doc.text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            # Determine end of the current chunk
            end = start + self.chunk_size

            # Extract chunk text
            chunk_text = text[start:end]

            # If not at the end of the string and we split inside a word,
            # try to backtrack to the last space to avoid breaking words
            if (
                end < text_length
                and not text[end - 1].isspace()
                and not text[end].isspace()
            ):
                last_space = chunk_text.rfind(" ")
                # Only backtrack if we found a space and it doesn't make the chunk too small  # noqa: E501
                if last_space > self.chunk_size // 2:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        filename=doc.filename,
                        source=doc.source,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            # Move start forward
            if end >= text_length:
                break

            # The next chunk starts `chunk_overlap` characters before the current end
            start = end - self.chunk_overlap

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Split multiple Documents into Chunks."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
