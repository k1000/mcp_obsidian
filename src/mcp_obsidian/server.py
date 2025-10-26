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
    ListNotesRequest,
    OperationResponse,
    OperationStatus,
    SearchQuery,
    UpdateNoteRequest,
)
from .obsidian_utils import ObsidianParser
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

        # Create MCP server
        self.mcp = FastMCP("obsidian-vault")

        # Register tools
        self._register_tools()

        logger.info("ObsidianMCPServer initialized")

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
                path = InputValidator.sanitize_filename(path)

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
