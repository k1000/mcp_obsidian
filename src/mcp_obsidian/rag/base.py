"""
Abstract base classes for RAG components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Name of the provider (e.g., "ollama", "openai")
        """
        pass


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: Document IDs
            embeddings: Document embeddings
            documents: Document texts
            metadatas: Optional metadata for each document
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of tuples (id, score, document, metadata)
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the store.

        Args:
            doc_id: Document ID to delete
        """
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete multiple documents from the store.

        Args:
            doc_ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents from the store."""
        pass

    @abstractmethod
    async def get_count(self) -> int:
        """
        Get total number of documents in the store.

        Returns:
            Document count
        """
        pass

    @abstractmethod
    async def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists.

        Args:
            doc_id: Document ID

        Returns:
            True if document exists
        """
        pass
