"""
FastMCP server for Obsidian vault access.
"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .auth import ApiKeyAuth, InputValidator, RateLimiter
from .config import get_config
from .models import (
    CreateNoteRequest,
    IndexOperation,
    ListNotesRequest,
    OperationResponse,
    OperationStatus,
    SemanticSearchQuery,
    SearchQuery,
    UpdateNoteRequest,
)
from .obsidian_utils import ObsidianParser
from .rag import (
    ChromaVectorStore,
    DocumentChunker,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    RAGEngine,
)
from .search import SearchEngine
from .vault import VaultManager, VaultOperationError

logger = logging.getLogger(__name__)


class ObsidianMCPServer:
    """MCP Server for Obsidian vault operations."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Obsidian MCP Server.

        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = get_config(config_path)

        # Initialize vault manager
        self.vault_manager = VaultManager(
            vault_path=self.config.vault.path,
            max_file_size_mb=self.config.vault.max_file_size_mb,
            allowed_extensions=self.config.vault.allowed_extensions,
        )

        # Initialize search engine
        self.search_engine = SearchEngine(
            vault_manager=self.vault_manager,
            cache_enabled=self.config.search.cache_enabled,
        )

        # Initialize authentication
        self.auth = None
        self.rate_limiter = None
        if self.config.auth.enabled:
            self.auth = ApiKeyAuth(self.config.auth.api_keys)
            if self.config.auth.rate_limit.enabled:
                self.rate_limiter = RateLimiter(
                    requests_per_minute=self.config.auth.rate_limit.requests_per_minute,
                    burst_size=self.config.auth.rate_limit.burst_size,
                )

        # Initialize RAG engine if enabled
        self.rag_engine = None
        if self.config.rag.enabled:
            self.rag_engine = self._initialize_rag_engine()
            logger.info("RAG engine initialized")

        # Create MCP server
        self.mcp = FastMCP("obsidian-vault")

        # Register tools
        self._register_tools()

        logger.info("ObsidianMCPServer initialized")

    def _initialize_rag_engine(self) -> RAGEngine:
        """Initialize RAG engine with configured provider."""
        # Initialize embedding provider based on config
        provider_name = self.config.rag.provider.lower()

        if provider_name == "ollama":
            embedding_provider = OllamaEmbeddingProvider(
                base_url=self.config.rag.providers.ollama.base_url,
                model=self.config.rag.providers.ollama.model,
                timeout=self.config.rag.providers.ollama.timeout,
            )
        elif provider_name == "openai":
            embedding_provider = OpenAIEmbeddingProvider(
                api_key=self.config.rag.providers.openai.api_key,
                model=self.config.rag.providers.openai.model,
                dimensions=self.config.rag.providers.openai.dimensions,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")

        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory=self.config.rag.vector_db_path,
            collection_name="obsidian",
        )

        # Initialize chunker
        chunker = DocumentChunker(
            chunk_size=self.config.rag.chunking.chunk_size,
            chunk_overlap=self.config.rag.chunking.chunk_overlap,
            strategy=self.config.rag.chunking.strategy,
            split_on_headers=self.config.rag.chunking.split_on_headers,
        )

        # Create RAG engine
        rag_engine = RAGEngine(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            vault_manager=self.vault_manager,
            chunker=chunker,
            cache_embeddings=self.config.rag.cache_embeddings,
            batch_size=self.config.rag.batch_size,
        )

        return rag_engine

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.mcp.tool()
        async def read_note(path: str) -> str:
            """
            Read a note from the Obsidian vault.

            Args:
                path: Relative path to the note within the vault

            Returns:
                JSON string containing note data with metadata, content, frontmatter, tags, and links
            """
            try:
                note = await self.vault_manager.read_note(path)
                return note.model_dump_json(indent=2)
            except VaultOperationError as e:
                logger.error(f"Error reading note: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to read note: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def create_note(
            path: str, content: str, frontmatter: Optional[Dict[str, Any]] = None
        ) -> str:
            """
            Create a new note in the Obsidian vault.

            Args:
                path: Relative path for the new note (will add .md if not present)
                content: Content of the note in markdown format
                frontmatter: Optional dictionary of YAML frontmatter data

            Returns:
                JSON string with created note data or error message
            """
            try:
                # Validate inputs
                path = InputValidator.sanitize_note_path(path)

                note = await self.vault_manager.create_note(path, content, frontmatter)

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Note created successfully: {path}",
                    data=note.model_dump(),
                ).model_dump_json(indent=2)

            except VaultOperationError as e:
                logger.error(f"Error creating note: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to create note: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def update_note(
            path: str,
            content: Optional[str] = None,
            frontmatter: Optional[Dict[str, Any]] = None,
            append: bool = False,
        ) -> str:
            """
            Update an existing note in the vault.

            Args:
                path: Relative path to the note
                content: New content (or content to append if append=True)
                frontmatter: Frontmatter fields to update (will be merged with existing)
                append: If True, append content instead of replacing

            Returns:
                JSON string with updated note data or error message
            """
            try:
                note = await self.vault_manager.update_note(
                    path, content=content, frontmatter=frontmatter, append=append
                )

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Note updated successfully: {path}",
                    data=note.model_dump(),
                ).model_dump_json(indent=2)

            except VaultOperationError as e:
                logger.error(f"Error updating note: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to update note: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def delete_note(path: str) -> str:
            """
            Delete a note from the vault.

            Args:
                path: Relative path to the note

            Returns:
                JSON string with operation status
            """
            try:
                await self.vault_manager.delete_note(path)

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Note deleted successfully: {path}",
                ).model_dump_json(indent=2)

            except VaultOperationError as e:
                logger.error(f"Error deleting note: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to delete note: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def list_notes(
            path_pattern: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
            sort_by: str = "modified",
            sort_desc: bool = True,
        ) -> str:
            """
            List notes in the vault with optional filtering and sorting.

            Args:
                path_pattern: Optional glob pattern to filter paths (e.g., "folder/**/*.md")
                limit: Maximum number of notes to return (1-1000)
                offset: Offset for pagination
                sort_by: Sort by field: name, modified, created, size
                sort_desc: Sort in descending order

            Returns:
                JSON string with list of note metadata
            """
            try:
                notes = await self.vault_manager.list_notes(path_pattern=path_pattern)

                # Sort notes
                sort_key_map = {
                    "name": lambda n: n.name.lower(),
                    "modified": lambda n: n.modified,
                    "created": lambda n: n.created,
                    "size": lambda n: n.size,
                }

                if sort_by in sort_key_map:
                    notes.sort(key=sort_key_map[sort_by], reverse=sort_desc)

                # Apply pagination
                total = len(notes)
                notes = notes[offset : offset + limit]

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Found {total} notes",
                    data={"notes": [n.model_dump() for n in notes], "total": total},
                ).model_dump_json(indent=2)

            except Exception as e:
                logger.error(f"Error listing notes: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to list notes: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def search_notes(
            query: str,
            tags: Optional[List[str]] = None,
            path_pattern: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
            case_sensitive: bool = False,
        ) -> str:
            """
            Search through vault content.

            Args:
                query: Search query string
                tags: Optional list of tags to filter by
                path_pattern: Optional path pattern to filter results
                limit: Maximum number of results (1-1000)
                offset: Offset for pagination
                case_sensitive: Whether search is case-sensitive

            Returns:
                JSON string with search results including matches and scores
            """
            try:
                search_query = SearchQuery(
                    query=query,
                    tags=tags,
                    path_pattern=path_pattern,
                    limit=limit,
                    offset=offset,
                    case_sensitive=case_sensitive,
                )

                results = await self.search_engine.search(search_query)

                return results.model_dump_json(indent=2)

            except Exception as e:
                logger.error(f"Error searching notes: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Search failed: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def get_backlinks(note_name: str) -> str:
            """
            Find all notes that link to a specific note (backlinks).

            Args:
                note_name: Name of the note to find backlinks for

            Returns:
                JSON string with list of note paths that link to this note
            """
            try:
                backlinks = await self.search_engine.search_by_links(note_name)

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Found {len(backlinks)} backlinks",
                    data={"backlinks": backlinks},
                ).model_dump_json(indent=2)

            except Exception as e:
                logger.error(f"Error getting backlinks: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to get backlinks: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def get_vault_stats() -> str:
            """
            Get statistics about the vault.

            Returns:
                JSON string with vault statistics including total notes, size, tags, etc.
            """
            try:
                stats = await self.vault_manager.get_vault_stats()
                return stats.model_dump_json(indent=2)

            except Exception as e:
                logger.error(f"Error getting vault stats: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to get vault stats: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def add_tag_to_note(path: str, tag: str, to_frontmatter: bool = True) -> str:
            """
            Add a tag to a note.

            Args:
                path: Relative path to the note
                tag: Tag to add (without # prefix)
                to_frontmatter: If True, add to frontmatter; otherwise append inline

            Returns:
                JSON string with operation status
            """
            try:
                # Read current note
                note = await self.vault_manager.read_note(path)

                # Add tag to content
                new_content = ObsidianParser.add_tag(
                    note.content, tag, to_frontmatter=to_frontmatter
                )

                # Update note
                await self.vault_manager.update_note(path, content=new_content)

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Tag '{tag}' added to note: {path}",
                ).model_dump_json(indent=2)

            except Exception as e:
                logger.error(f"Error adding tag: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to add tag: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        @self.mcp.tool()
        async def remove_tag_from_note(path: str, tag: str) -> str:
            """
            Remove a tag from a note.

            Args:
                path: Relative path to the note
                tag: Tag to remove (without # prefix)

            Returns:
                JSON string with operation status
            """
            try:
                # Read current note
                note = await self.vault_manager.read_note(path)

                # Remove tag from content
                new_content = ObsidianParser.remove_tag(note.content, tag)

                # Update note
                await self.vault_manager.update_note(path, content=new_content)

                return OperationResponse(
                    status=OperationStatus.SUCCESS,
                    message=f"Tag '{tag}' removed from note: {path}",
                ).model_dump_json(indent=2)

            except Exception as e:
                logger.error(f"Error removing tag: {e}")
                return OperationResponse(
                    status=OperationStatus.ERROR,
                    message=f"Failed to remove tag: {str(e)}",
                    error=str(e),
                ).model_dump_json()

        # RAG/Semantic Search Tools
        if self.rag_engine:

            @self.mcp.tool()
            async def semantic_search(
                query: str,
                k: int = 10,
                tags: Optional[List[str]] = None,
                path_pattern: Optional[str] = None,
            ) -> str:
                """
                Perform semantic search across vault notes using embeddings.

                Args:
                    query: Search query text
                    k: Number of results to return (1-100)
                    tags: Optional list of tags to filter by
                    path_pattern: Optional path pattern to filter results

                Returns:
                    JSON string with semantic search results including similarity scores
                """
                try:
                    results = await self.rag_engine.semantic_search(
                        query=query,
                        k=min(k, 100),
                        tags=tags,
                        path_pattern=path_pattern,
                    )

                    from .models import SemanticSearchResponse
                    import time

                    response = SemanticSearchResponse(
                        results=results,
                        total=len(results),
                        query=query,
                        duration_ms=0.0,  # Already calculated in engine
                        hybrid_mode=False,  # Will be updated when hybrid search is implemented
                    )

                    return response.model_dump_json(indent=2)

                except Exception as e:
                    logger.error(f"Error performing semantic search: {e}")
                    return OperationResponse(
                        status=OperationStatus.ERROR,
                        message=f"Semantic search failed: {str(e)}",
                        error=str(e),
                    ).model_dump_json()

            @self.mcp.tool()
            async def index_vault(
                force_reindex: bool = False,
                path_pattern: Optional[str] = None,
                batch_size: int = 32,
            ) -> str:
                """
                Index vault notes for semantic search.

                Args:
                    force_reindex: Force re-indexing of all documents
                    path_pattern: Optional path pattern to index only matching notes
                    batch_size: Batch size for processing (1-100)

                Returns:
                    JSON string with indexing statistics
                """
                try:
                    notes_indexed, chunks_added, chunks_updated = await self.rag_engine.index_vault(
                        force_reindex=force_reindex,
                        path_pattern=path_pattern,
                    )

                    return OperationResponse(
                        status=OperationStatus.SUCCESS,
                        message=f"Indexing complete: {notes_indexed} notes indexed",
                        data={
                            "notes_indexed": notes_indexed,
                            "chunks_added": chunks_added,
                            "chunks_updated": chunks_updated,
                        },
                    ).model_dump_json(indent=2)

                except Exception as e:
                    logger.error(f"Error indexing vault: {e}")
                    return OperationResponse(
                        status=OperationStatus.ERROR,
                        message=f"Indexing failed: {str(e)}",
                        error=str(e),
                    ).model_dump_json()

            @self.mcp.tool()
            async def get_index_stats() -> str:
                """
                Get statistics about the semantic search index.

                Returns:
                    JSON string with index statistics including document count, size, and provider info
                """
                try:
                    stats = await self.rag_engine.get_index_stats()
                    return stats.model_dump_json(indent=2)

                except Exception as e:
                    logger.error(f"Error getting index stats: {e}")
                    return OperationResponse(
                        status=OperationStatus.ERROR,
                        message=f"Failed to get index stats: {str(e)}",
                        error=str(e),
                    ).model_dump_json()

            @self.mcp.tool()
            async def delete_index() -> str:
                """
                Clear the entire semantic search index.

                WARNING: This will delete all indexed embeddings and cannot be undone.

                Returns:
                    JSON string with operation status
                """
                try:
                    await self.rag_engine.clear_index()

                    return OperationResponse(
                        status=OperationStatus.SUCCESS,
                        message="Index cleared successfully",
                    ).model_dump_json(indent=2)

                except Exception as e:
                    logger.error(f"Error clearing index: {e}")
                    return OperationResponse(
                        status=OperationStatus.ERROR,
                        message=f"Failed to clear index: {str(e)}",
                        error=str(e),
                    ).model_dump_json()

    def run(self) -> None:
        """Run the MCP server."""
        logger.info("Starting Obsidian MCP Server...")
        self.mcp.run()


def main() -> None:
    """Main entry point."""
    import sys

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get config path from command line args
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Create and run server
    server = ObsidianMCPServer(config_path=config_path)
    server.run()


if __name__ == "__main__":
    main()
