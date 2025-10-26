"""
Core vault operations for reading, writing, and managing Obsidian notes.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import aiofiles.os

from .auth import InputValidator, PathValidator
from .models import Note, NoteMetadata, VaultStats
from .obsidian_utils import ObsidianParser

logger = logging.getLogger(__name__)


class VaultOperationError(Exception):
    """Exception raised for vault operation errors."""

    pass


class VaultManager:
    """Manages all operations on an Obsidian vault."""

    def __init__(
        self,
        vault_path: str,
        max_file_size_mb: int = 10,
        allowed_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize vault manager.

        Args:
            vault_path: Path to the Obsidian vault
            max_file_size_mb: Maximum file size in MB
            allowed_extensions: List of allowed file extensions
        """
        self.vault_path = Path(vault_path).resolve()
        self.max_file_size_mb = max_file_size_mb
        self.allowed_extensions = allowed_extensions or ["md"]

        # Lock for concurrent operations
        self._locks: Dict[str, asyncio.Lock] = {}

        if not self.vault_path.exists():
            raise VaultOperationError(f"Vault path does not exist: {vault_path}")

        logger.info(f"Initialized VaultManager for: {self.vault_path}")

    def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for a specific file path."""
        if path not in self._locks:
            self._locks[path] = asyncio.Lock()
        return self._locks[path]

    def _validate_path(self, relative_path: str) -> Path:
        """
        Validate and resolve a relative path.

        Args:
            relative_path: Path relative to vault root

        Returns:
            Resolved absolute path

        Raises:
            VaultOperationError: If path is invalid
        """
        if not PathValidator.is_safe_path(self.vault_path, relative_path):
            raise VaultOperationError(f"Invalid or unsafe path: {relative_path}")

        return self.vault_path / relative_path

    async def read_note(self, path: str) -> Note:
        """
        Read a note from the vault.

        Args:
            path: Relative path to the note

        Returns:
            Note object with metadata and content

        Raises:
            VaultOperationError: If note cannot be read
        """
        file_path = self._validate_path(path)

        if not await aiofiles.os.path.exists(file_path):
            raise VaultOperationError(f"Note not found: {path}")

        async with self._get_lock(path):
            try:
                # Read file content
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()

                # Get file stats
                stat = await aiofiles.os.stat(file_path)

                # Parse Obsidian features
                frontmatter_dict, _ = ObsidianParser.extract_frontmatter(content)
                tags = ObsidianParser.extract_tags(content)
                links = ObsidianParser.extract_wiki_links(content)

                # Create metadata
                metadata = NoteMetadata(
                    path=path,
                    name=file_path.name,
                    size=stat.st_size,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    modified=datetime.fromtimestamp(stat.st_mtime),
                    extension=file_path.suffix.lstrip("."),
                )

                # Parse frontmatter into Pydantic model if exists
                from .models import Frontmatter

                frontmatter = None
                if frontmatter_dict:
                    frontmatter = Frontmatter(**frontmatter_dict)

                note = Note(
                    metadata=metadata,
                    content=content,
                    frontmatter=frontmatter,
                    tags=tags,
                    links=links,
                    backlinks=[],  # Backlinks computed separately
                )

                logger.debug(f"Successfully read note: {path}")
                return note

            except Exception as e:
                logger.error(f"Error reading note {path}: {e}")
                raise VaultOperationError(f"Failed to read note: {e}")

    async def create_note(
        self, path: str, content: str, frontmatter: Optional[Dict] = None
    ) -> Note:
        """
        Create a new note in the vault.

        Args:
            path: Relative path for the new note
            content: Note content
            frontmatter: Optional frontmatter data

        Returns:
            Created Note object

        Raises:
            VaultOperationError: If note cannot be created
        """
        # Ensure path ends with .md
        if not path.endswith(".md"):
            path = f"{path}.md"

        file_path = self._validate_path(path)

        # Check if file already exists
        if await aiofiles.os.path.exists(file_path):
            raise VaultOperationError(f"Note already exists: {path}")

        # Validate content size
        if not InputValidator.validate_content_size(content, self.max_file_size_mb):
            raise VaultOperationError(
                f"Content exceeds maximum size of {self.max_file_size_mb}MB"
            )

        async with self._get_lock(path):
            try:
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Add frontmatter if provided
                if frontmatter:
                    content = ObsidianParser.add_frontmatter(content, frontmatter)

                # Write file
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(content)

                logger.info(f"Created note: {path}")

                # Read and return the created note
                return await self.read_note(path)

            except Exception as e:
                logger.error(f"Error creating note {path}: {e}")
                raise VaultOperationError(f"Failed to create note: {e}")

    async def update_note(
        self,
        path: str,
        content: Optional[str] = None,
        frontmatter: Optional[Dict] = None,
        append: bool = False,
    ) -> Note:
        """
        Update an existing note.

        Args:
            path: Relative path to the note
            content: New content (or content to append)
            frontmatter: Frontmatter updates
            append: If True, append content instead of replacing

        Returns:
            Updated Note object

        Raises:
            VaultOperationError: If note cannot be updated
        """
        file_path = self._validate_path(path)

        if not await aiofiles.os.path.exists(file_path):
            raise VaultOperationError(f"Note not found: {path}")

        async with self._get_lock(path):
            try:
                # Read current content
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    current_content = await f.read()

                new_content = current_content

                # Update frontmatter
                if frontmatter:
                    new_content = ObsidianParser.update_frontmatter(
                        new_content, frontmatter, merge=True
                    )

                # Update content
                if content:
                    if append:
                        # Append to existing content
                        new_content = f"{new_content.rstrip()}\n\n{content}"
                    else:
                        # Replace content (preserve frontmatter if exists)
                        existing_fm, _ = ObsidianParser.extract_frontmatter(new_content)
                        if existing_fm:
                            new_content = ObsidianParser.add_frontmatter(content, existing_fm)
                        else:
                            new_content = content

                # Validate content size
                if not InputValidator.validate_content_size(
                    new_content, self.max_file_size_mb
                ):
                    raise VaultOperationError(
                        f"Content exceeds maximum size of {self.max_file_size_mb}MB"
                    )

                # Write updated content
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(new_content)

                logger.info(f"Updated note: {path}")

                # Read and return the updated note
                return await self.read_note(path)

            except Exception as e:
                logger.error(f"Error updating note {path}: {e}")
                raise VaultOperationError(f"Failed to update note: {e}")

    async def delete_note(self, path: str) -> bool:
        """
        Delete a note from the vault.

        Args:
            path: Relative path to the note

        Returns:
            True if deleted successfully

        Raises:
            VaultOperationError: If note cannot be deleted
        """
        file_path = self._validate_path(path)

        if not await aiofiles.os.path.exists(file_path):
            raise VaultOperationError(f"Note not found: {path}")

        async with self._get_lock(path):
            try:
                await aiofiles.os.remove(file_path)
                logger.info(f"Deleted note: {path}")
                return True

            except Exception as e:
                logger.error(f"Error deleting note {path}: {e}")
                raise VaultOperationError(f"Failed to delete note: {e}")

    async def list_notes(
        self,
        path_pattern: Optional[str] = None,
        extension: str = "md",
    ) -> List[NoteMetadata]:
        """
        List all notes in the vault.

        Args:
            path_pattern: Optional glob pattern to filter paths
            extension: File extension to filter (default: md)

        Returns:
            List of note metadata
        """
        notes = []

        try:
            # Use pathlib's rglob for recursive search
            pattern = f"**/*.{extension}" if not path_pattern else path_pattern

            for file_path in self.vault_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        stat = await aiofiles.os.stat(file_path)
                        relative_path = file_path.relative_to(self.vault_path)

                        metadata = NoteMetadata(
                            path=str(relative_path),
                            name=file_path.name,
                            size=stat.st_size,
                            created=datetime.fromtimestamp(stat.st_ctime),
                            modified=datetime.fromtimestamp(stat.st_mtime),
                            extension=file_path.suffix.lstrip("."),
                        )

                        notes.append(metadata)

                    except Exception as e:
                        logger.warning(f"Error processing file {file_path}: {e}")
                        continue

            logger.debug(f"Listed {len(notes)} notes")
            return notes

        except Exception as e:
            logger.error(f"Error listing notes: {e}")
            raise VaultOperationError(f"Failed to list notes: {e}")

    async def get_vault_stats(self) -> VaultStats:
        """
        Get statistics about the vault.

        Returns:
            VaultStats object with vault information
        """
        try:
            notes = await self.list_notes()

            total_size = sum(note.size for note in notes)
            last_modified = max((note.modified for note in notes), default=datetime.now())

            # Count by extension
            ext_counts: Dict[str, int] = {}
            for note in notes:
                ext_counts[note.extension] = ext_counts.get(note.extension, 0) + 1

            # Collect all unique tags
            all_tags = set()
            for note in notes:
                try:
                    note_obj = await self.read_note(note.path)
                    all_tags.update(note_obj.tags)
                except Exception:
                    continue

            return VaultStats(
                total_notes=len(notes),
                total_size_bytes=total_size,
                total_tags=len(all_tags),
                unique_tags=sorted(list(all_tags)),
                last_modified=last_modified,
                note_count_by_extension=ext_counts,
            )

        except Exception as e:
            logger.error(f"Error getting vault stats: {e}")
            raise VaultOperationError(f"Failed to get vault stats: {e}")
