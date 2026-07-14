"""Document loading module.

Loads documents from local directories or S3, supporting text, markdown, and
the specific Syncora HR dataset JSONL format.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A raw document loaded from a source before chunking."""

    text: str
    filename: str
    source: str


class DocumentLoader:
    """Loads documents from various sources."""

    def __init__(
        self, s3_bucket: str | None = None, aws_region: str | None = None
    ) -> None:  # noqa: E501
        """Initialize the loader.

        Args:
            s3_bucket: S3 bucket name for loading remote documents.
            aws_region: AWS region for S3 client.
        """
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        self._s3_client: Any = None

    @property
    def s3_client(self) -> Any:
        """Lazy-loaded S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client("s3", region_name=self.aws_region)
        return self._s3_client

    def _parse_jsonl_line(
        self, line: str, filename: str, line_number: int
    ) -> Document | None:  # noqa: E501
        """Parse a single JSONL line containing a ChatML conversation."""
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            if not messages:
                return None

            # Extract user and assistant turns
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )  # noqa: E501
            assistant_msg = next(
                (m["content"] for m in messages if m["role"] == "assistant"), None
            )  # noqa: E501

            if not user_msg or not assistant_msg:
                return None

            text = f"Q: {user_msg}\nA: {assistant_msg}"
            return Document(
                text=text,
                filename=filename,
                source=f"{filename}:{line_number}",
            )
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON at {filename}:{line_number}")
            return None

    def load_local_directory(self, directory: str | Path) -> Iterator[Document]:
        """Load documents from a local directory.

        Yields Document objects.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        for filepath in dir_path.rglob("*"):
            if not filepath.is_file():
                continue

            if filepath.suffix == ".jsonl":
                with open(filepath, encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        doc = self._parse_jsonl_line(line, filepath.name, i)
                        if doc:
                            yield doc
            elif filepath.suffix in (".txt", ".md"):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        yield Document(
                            text=text,
                            filename=filepath.name,
                            source=str(filepath),
                        )
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode text file: {filepath}")

    def load_s3_prefix(self, prefix: str) -> Iterator[Document]:
        """Load documents from an S3 bucket prefix."""
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                content = response["Body"].read().decode("utf-8")

                if key.endswith(".jsonl"):
                    for i, line in enumerate(content.splitlines(), 1):
                        line = line.strip()
                        if not line:
                            continue
                        # SEC-005: M1 content validation implementation starts here by
                        # handling the specific formats cleanly.
                        doc = self._parse_jsonl_line(line, key.split("/")[-1], i)
                        if doc:
                            yield doc
                elif key.endswith(".txt") or key.endswith(".md"):
                    if content.strip():
                        yield Document(
                            text=content,
                            filename=key.split("/")[-1],
                            source=f"s3://{self.s3_bucket}/{key}",
                        )
