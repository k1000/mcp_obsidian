"""
RAG Engine for semantic search and retrieval.
"""

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import IndexStats, NoteMetadata, SemanticSearchResult
from ..vault import VaultManager
from .base import EmbeddingProvider, VectorStore
from .chunking import DocumentChunker

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine for semantic search and document retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        vault_manager: VaultManager,
        chunker: DocumentChunker,
        cache_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize RAG engine.

        Args:
            embedding_provider: Embedding provider instance
            vector_store: Vector store instance
            vault_manager: Vault manager instance
            chunker: Document chunker instance
            cache_embeddings: Whether to cache embeddings
            batch_size: Batch size for embedding generation
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.vault_manager = vault_manager
        self.chunker = chunker
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size

        # Embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}

        logger.info(f"Initialized RAGEngine with provider '{embedding_provider.get_provider_name()}'")

    async def index_vault(
        self,
        force_reindex: bool = False,
        path_pattern: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Index all notes in the vault.

        Args:
            force_reindex: Force re-indexing of all documents
            path_pattern: Optional path pattern to filter notes

        Returns:
            Tuple of (notes_indexed, chunks_added, chunks_updated)
        """
        start_time = time.time()
        logger.info(f"Starting vault indexing (force_reindex={force_reindex})")

        # Get all notes
        notes = await self.vault_manager.list_notes(path_pattern=path_pattern)
        logger.info(f"Found {len(notes)} notes to index")

        notes_indexed = 0
        chunks_added = 0
        chunks_updated = 0

        # Process notes in batches
        for note_metadata in notes:
            try:
                result = await self._index_note(note_metadata, force_reindex)
                if result:
                    added, updated = result
                    notes_indexed += 1
                    chunks_added += added
                    chunks_updated += updated

            except Exception as e:
                logger.error(f"Error indexing note {note_metadata.path}: {e}")
                continue

        duration = time.time() - start_time
        logger.info(
            f"Indexing complete: {notes_indexed} notes, "
            f"{chunks_added} chunks added, {chunks_updated} chunks updated "
            f"in {duration:.2f}s"
        )

        return notes_indexed, chunks_added, chunks_updated

    async def _index_note(
        self, note_metadata: NoteMetadata, force_reindex: bool = False
    ) -> Optional[Tuple[int, int]]:
        """
        Index a single note.

        Args:
            note_metadata: Note metadata
            force_reindex: Force re-indexing

        Returns:
            Tuple of (chunks_added, chunks_updated) or None if skipped
        """
        # Read note content
        note = await self.vault_manager.read_note(note_metadata.path)

        # Chunk the document
        chunks = self.chunker.chunk_document(
            note.content,
            metadata={
                "note_path": note_metadata.path,
                "note_name": note_metadata.name,
                "modified": note_metadata.modified.isoformat(),
                "tags": ",".join(note.tags) if note.tags else "",  # Convert list to comma-separated string
            },
        )

        if not chunks:
            logger.debug(f"No chunks generated for {note_metadata.path}")
            return None

        chunks_added = 0
        chunks_updated = 0

        # Process chunks in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_texts = [chunk_text for chunk_text, _ in batch]
            batch_metadatas = [chunk_metadata for _, chunk_metadata in batch]

            # Generate embeddings
            try:
                embeddings = await self._get_embeddings_with_cache(batch_texts)
            except Exception as e:
                logger.error(f"Error generating embeddings for {note_metadata.path}: {e}")
                continue

            # Generate document IDs
            doc_ids = []
            for chunk_text, chunk_metadata in batch:
                chunk_idx = chunk_metadata.get("chunk_index", 0)
                doc_id = self._generate_doc_id(note_metadata.path, chunk_idx)
                doc_ids.append(doc_id)

            # Check which documents need to be updated
            ids_to_add = []
            embeddings_to_add = []
            texts_to_add = []
            metadatas_to_add = []

            for j, doc_id in enumerate(doc_ids):
                exists = await self.vector_store.document_exists(doc_id)
                if force_reindex or not exists:
                    ids_to_add.append(doc_id)
                    embeddings_to_add.append(embeddings[j])
                    texts_to_add.append(batch_texts[j])
                    metadatas_to_add.append(batch_metadatas[j])

                    if exists:
                        chunks_updated += 1
                    else:
                        chunks_added += 1

            # Add documents to vector store
            if ids_to_add:
                # Delete existing documents first if force_reindex
                if force_reindex:
                    await self.vector_store.delete_documents(ids_to_add)

                await self.vector_store.add_documents(
                    ids=ids_to_add,
                    embeddings=embeddings_to_add,
                    documents=texts_to_add,
                    metadatas=metadatas_to_add,
                )

        logger.debug(
            f"Indexed {note_metadata.path}: "
            f"{chunks_added} added, {chunks_updated} updated"
        )

        return chunks_added, chunks_updated

    async def semantic_search(
        self,
        query: str,
        k: int = 10,
        tags: Optional[List[str]] = None,
        path_pattern: Optional[str] = None,
    ) -> List[SemanticSearchResult]:
        """
        Perform semantic search.

        Args:
            query: Search query
            k: Number of results to return
            tags: Optional tags filter
            path_pattern: Optional path pattern filter

        Returns:
            List of search results
        """
        start_time = time.time()

        # Generate query embedding
        try:
            query_embedding = await self.embedding_provider.embed_text(query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}") from e

        # Search vector store (no pre-filtering, we'll filter results in Python)
        try:
            # Get extra results if we need to filter
            fetch_k = k * 3 if (tags or path_pattern) else k
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=fetch_k,
                filter=None,  # Do filtering post-retrieval for flexibility
            )
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise RuntimeError(f"Search failed: {e}") from e

        # Format results
        formatted_results = []
        for doc_id, score, chunk_text, metadata in results:
            try:
                note_path = metadata.get("note_path", "")
                note_name = metadata.get("note_name", "")
                chunk_index = metadata.get("chunk_index", 0)
                note_tags_str = metadata.get("tags", "")

                # Parse tags from comma-separated string
                note_tags = [t.strip() for t in note_tags_str.split(",") if t.strip()]

                # Apply tag filter if needed
                if tags:
                    # Check if any of the requested tags are in the note's tags
                    if not any(tag in note_tags for tag in tags):
                        continue

                # Apply path pattern filter if needed
                if path_pattern:
                    # Simple glob-like matching
                    if not self._match_path_pattern(note_path, path_pattern):
                        continue

                # Get note metadata
                note_metadata = NoteMetadata(
                    path=note_path,
                    name=note_name,
                    size=0,  # We don't have size readily available
                    created=datetime.now(),  # Placeholder
                    modified=datetime.fromisoformat(metadata.get("modified", datetime.now().isoformat())),
                )

                result = SemanticSearchResult(
                    note=note_metadata,
                    score=score,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                )

                formatted_results.append(result)

                if len(formatted_results) >= k:
                    break

            except Exception as e:
                logger.warning(f"Error formatting result: {e}")
                continue

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Semantic search completed: {len(formatted_results)} results in {duration_ms:.2f}ms")

        return formatted_results

    async def delete_note_from_index(self, note_path: str) -> int:
        """
        Delete all chunks of a note from the index.

        Args:
            note_path: Path to the note

        Returns:
            Number of chunks deleted
        """
        # Find all document IDs for this note
        # This is a simple approach - we'll try common chunk indices
        doc_ids = []
        for i in range(100):  # Assume max 100 chunks per note
            doc_id = self._generate_doc_id(note_path, i)
            if await self.vector_store.document_exists(doc_id):
                doc_ids.append(doc_id)
            elif i > 10:  # Stop if we haven't found anything in the last 10
                break

        if doc_ids:
            await self.vector_store.delete_documents(doc_ids)
            logger.info(f"Deleted {len(doc_ids)} chunks for note {note_path}")

        return len(doc_ids)

    async def get_index_stats(self) -> IndexStats:
        """
        Get statistics about the index.

        Returns:
            Index statistics
        """
        total_docs = await self.vector_store.get_count()
        embedding_dim = self.embedding_provider.get_dimension()
        provider_name = self.embedding_provider.get_provider_name()

        # Estimate number of unique notes (rough estimate)
        # In practice, we'd need to track this separately
        estimated_notes = total_docs // 5  # Assume avg 5 chunks per note

        # Get index size if available
        index_size_mb = 0.0
        if hasattr(self.vector_store, "get_size_mb"):
            index_size_mb = self.vector_store.get_size_mb()

        return IndexStats(
            total_documents=total_docs,
            total_notes=estimated_notes,
            embedding_dimension=embedding_dim,
            provider=provider_name,
            last_indexed=datetime.now(),
            index_size_mb=index_size_mb,
        )

    async def clear_index(self) -> None:
        """Clear the entire index."""
        await self.vector_store.clear()
        self._embedding_cache.clear()
        logger.info("Index cleared")

    async def _get_embeddings_with_cache(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not self.cache_embeddings:
            return await self.embedding_provider.embed_batch(texts)

        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                embeddings.append(self._embedding_cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Generate missing embeddings
        if texts_to_embed:
            new_embeddings = await self.embedding_provider.embed_batch(texts_to_embed)

            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
                cache_key = self._get_cache_key(texts[idx])
                self._embedding_cache[cache_key] = embedding

        return embeddings  # type: ignore

    def _generate_doc_id(self, note_path: str, chunk_index: int) -> str:
        """Generate a unique document ID."""
        return f"{note_path}::chunk_{chunk_index}"

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _match_path_pattern(self, path: str, pattern: str) -> bool:
        """Simple path pattern matching."""
        # Basic implementation - can be enhanced with proper glob matching
        if "*" in pattern:
            pattern_parts = pattern.split("*")
            pos = 0
            for part in pattern_parts:
                if part:
                    idx = path.find(part, pos)
                    if idx == -1:
                        return False
                    pos = idx + len(part)
            return True
        else:
            return pattern in path
