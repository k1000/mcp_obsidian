"""
Vector store implementations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from .base import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, persist_directory: str = "data/vector_db", collection_name: str = "obsidian"):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            f"Initialized ChromaVectorStore at '{persist_directory}' "
            f"with collection '{collection_name}'"
        )

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
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas if metadatas else None,
            )
            logger.debug(f"Added {len(ids)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise RuntimeError(f"Failed to add documents: {e}") from e

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
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter if filter else None,
            )

            # Format results
            formatted_results = []
            if results and results.get("ids"):
                for i in range(len(results["ids"][0])):
                    doc_id = results["ids"][0][i]
                    distance = results["distances"][0][i] if "distances" in results else 0.0
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance
                    document = results["documents"][0][i] if "documents" in results else ""
                    metadata = results["metadatas"][0][i] if "metadatas" in results else {}

                    formatted_results.append((doc_id, score, document, metadata))

            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise RuntimeError(f"Search failed: {e}") from e

    async def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the store.

        Args:
            doc_id: Document ID to delete
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document '{doc_id}'")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise RuntimeError(f"Failed to delete document: {e}") from e

    async def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete multiple documents from the store.

        Args:
            doc_ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=doc_ids)
            logger.debug(f"Deleted {len(doc_ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise RuntimeError(f"Failed to delete documents: {e}") from e

    async def clear(self) -> None:
        """Clear all documents from the store."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared all documents from collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise RuntimeError(f"Failed to clear vector store: {e}") from e

    async def get_count(self) -> int:
        """
        Get total number of documents in the store.

        Returns:
            Document count
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    async def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists.

        Args:
            doc_id: Document ID

        Returns:
            True if document exists
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False

    def get_size_mb(self) -> float:
        """
        Get approximate size of the vector database in MB.

        Returns:
            Size in megabytes
        """
        try:
            total_size = 0
            db_path = Path(self.persist_directory)
            if db_path.exists():
                for file_path in db_path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Error calculating database size: {e}")
            return 0.0
