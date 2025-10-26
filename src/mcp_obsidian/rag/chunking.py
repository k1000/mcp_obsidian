"""
Document chunking utilities for RAG.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Document chunking for embedding generation."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "smart",
        split_on_headers: bool = True,
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            strategy: Chunking strategy (smart, fixed, recursive)
            split_on_headers: Whether to split on markdown headers
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.split_on_headers = split_on_headers

        logger.info(
            f"Initialized DocumentChunker with strategy='{strategy}', "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(self, content: str, metadata: dict | None = None) -> List[Tuple[str, dict]]:
        """
        Chunk a document into smaller pieces.

        Args:
            content: Document content
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        if self.strategy == "smart":
            return self._smart_chunk(content, metadata)
        elif self.strategy == "fixed":
            return self._fixed_chunk(content, metadata)
        elif self.strategy == "recursive":
            return self._recursive_chunk(content, metadata)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using smart chunking")
            return self._smart_chunk(content, metadata)

    def _smart_chunk(self, content: str, metadata: dict | None = None) -> List[Tuple[str, dict]]:
        """
        Smart chunking that respects markdown structure.

        Args:
            content: Document content
            metadata: Optional metadata

        Returns:
            List of chunks with metadata
        """
        chunks = []
        base_metadata = metadata or {}

        # If split_on_headers is enabled, first try to split by headers
        if self.split_on_headers:
            sections = self._split_by_headers(content)
        else:
            sections = [content]

        # Process each section
        for section_idx, section in enumerate(sections):
            if len(section) <= self.chunk_size:
                # Section fits in one chunk
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": len(chunks),
                    "section_index": section_idx,
                }
                chunks.append((section.strip(), chunk_metadata))
            else:
                # Section needs to be split further
                section_chunks = self._split_by_paragraphs(section)
                for chunk_text in section_chunks:
                    chunk_metadata = {
                        **base_metadata,
                        "chunk_index": len(chunks),
                        "section_index": section_idx,
                    }
                    chunks.append((chunk_text.strip(), chunk_metadata))

        logger.debug(f"Smart chunking produced {len(chunks)} chunks")
        return chunks

    def _fixed_chunk(self, content: str, metadata: dict | None = None) -> List[Tuple[str, dict]]:
        """
        Fixed-size chunking with overlap.

        Args:
            content: Document content
            metadata: Optional metadata

        Returns:
            List of chunks with metadata
        """
        chunks = []
        base_metadata = metadata or {}

        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            chunk_metadata = {**base_metadata, "chunk_index": chunk_idx}
            chunks.append((chunk_text.strip(), chunk_metadata))

            # Move start forward by chunk_size minus overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_idx += 1

        logger.debug(f"Fixed chunking produced {len(chunks)} chunks")
        return chunks

    def _recursive_chunk(self, content: str, metadata: dict | None = None) -> List[Tuple[str, dict]]:
        """
        Recursive chunking that tries multiple separators.

        Args:
            content: Document content
            metadata: Optional metadata

        Returns:
            List of chunks with metadata
        """
        separators = ["\n\n", "\n", ". ", " "]
        chunks = self._recursive_split(content, separators, metadata or {})

        logger.debug(f"Recursive chunking produced {len(chunks)} chunks")
        return chunks

    def _recursive_split(
        self, text: str, separators: List[str], base_metadata: dict, chunk_idx: int = 0
    ) -> List[Tuple[str, dict]]:
        """Recursively split text using different separators."""
        chunks = []

        if len(text) <= self.chunk_size:
            chunk_metadata = {**base_metadata, "chunk_index": chunk_idx}
            return [(text.strip(), chunk_metadata)]

        # Try each separator
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""

                for part in parts:
                    if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                        current_chunk += part + separator
                    else:
                        if current_chunk:
                            chunk_metadata = {**base_metadata, "chunk_index": len(chunks)}
                            chunks.append((current_chunk.strip(), chunk_metadata))
                        current_chunk = part + separator

                # Add remaining chunk
                if current_chunk:
                    chunk_metadata = {**base_metadata, "chunk_index": len(chunks)}
                    chunks.append((current_chunk.strip(), chunk_metadata))

                return chunks

        # If no separator works, do fixed chunking
        return self._fixed_chunk(text, base_metadata)

    def _split_by_headers(self, content: str) -> List[str]:
        """
        Split content by markdown headers.

        Args:
            content: Markdown content

        Returns:
            List of sections
        """
        # Match markdown headers (# Header, ## Header, etc.)
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections = []
        last_pos = 0

        for match in header_pattern.finditer(content):
            start = match.start()
            if start > last_pos:
                sections.append(content[last_pos:start])
            last_pos = start

        # Add remaining content
        if last_pos < len(content):
            sections.append(content[last_pos:])

        # Filter out empty sections
        sections = [s for s in sections if s.strip()]

        return sections if sections else [content]

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text by paragraphs with overlap.

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # Start new chunk with overlap if possible
                if len(para) > self.chunk_size:
                    # Paragraph is too long, split it
                    words = para.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                            current_chunk += word + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                else:
                    current_chunk = para + "\n\n"

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]
