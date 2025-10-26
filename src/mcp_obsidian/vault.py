"""
Core vault operations for reading, writing, and managing Obsidian notes.
"""

import asyncio
import logging
import os
import tempfile
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
        """Get or create a lock for a specific file path using normalized path."""
        # Normalize path for consistent locking
        normalized_path = InputValidator.sanitize_note_path(path)
        if normalized_path not in self._locks:
            self._locks[normalized_path] = asyncio.Lock()
        return self._locks[normalized_path]

    def _validate_path(self, relative_path: str) -> tuple[Path, str]:
        """
        Validate and resolve a relative path with normalization and extension checking.

        Args:
            relative_path: Path relative to vault root

        Returns:
            Tuple of (resolved absolute path, normalized relative path)

        Raises:
            VaultOperationError: If path is invalid
        """
        # Normalize the path first
        normalized_path = InputValidator.sanitize_note_path(relative_path)
        
        # Validate the normalized path is safe
        if not PathValidator.is_safe_path(self.vault_path, normalized_path):
            raise VaultOperationError(f"Invalid or unsafe path: {relative_path}")
        
        # Extract filename to check extension
        filename = Path(normalized_path).name
        if not InputValidator.validate_extension(filename, self.allowed_extensions):
            allowed_exts = ", ".join(self.allowed_extensions)
            raise VaultOperationError(
                f"File extension not allowed. Allowed extensions: {allowed_exts}. Got: {filename}"
            )

        resolved_path = self.vault_path / normalized_path
        return resolved_path, normalized_path

    async def _atomic_write(self, file_path: Path, content: str) -> None:
        """
        Atomically write content to a file using temp file + rename.
        
        Args:
            file_path: Target file path
            content: Content to write
            
        Raises:
            VaultOperationError: If write fails
        """
        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary file in the same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent,
                prefix=f".{file_path.name}.tmp",
                suffix=".md"
            )
            
            try:
                # Write content to temp file
                async with aiofiles.open(temp_fd, "w", encoding="utf-8", closefd=False) as f:
                    await f.write(content)
                    await f.flush()
                    # Force write to disk for durability
                    os.fsync(temp_fd)
                
                # Close the file descriptor
                os.close(temp_fd)
                
                # Atomically replace the target file
                os.replace(temp_path, file_path)
                
            except Exception:
                # Clean up temp file on error
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
                
        except Exception as e:
            raise VaultOperationError(f"Failed to write file atomically: {e}")

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
        file_path, normalized_path = self._validate_path(path)

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

                # Create metadata using normalized path
                metadata = NoteMetadata(
                    path=normalized_path,
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

                logger.debug(f"Successfully read note: {normalized_path}")
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
        # Normalize and validate path (extension will be added if missing)
        file_path, normalized_path = self._validate_path(path)

        # Check if file already exists
        if await aiofiles.os.path.exists(file_path):
            raise VaultOperationError(f"Note already exists: {normalized_path}")

        # Validate content size
        if not InputValidator.validate_content_size(content, self.max_file_size_mb):
            raise VaultOperationError(
                f"Content exceeds maximum size of {self.max_file_size_mb}MB"
            )

        async with self._get_lock(path):
            try:
                # Add frontmatter if provided
                final_content = content
                if frontmatter:
                    final_content = ObsidianParser.add_frontmatter(content, frontmatter)

                # Write file atomically
                await self._atomic_write(file_path, final_content)

                logger.info(f"Created note: {normalized_path}")

                # Read and return the created note
                return await self.read_note(normalized_path)

            except Exception as e:
                logger.error(f"Error creating note {normalized_path}: {e}")
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
        file_path, normalized_path = self._validate_path(path)

        if not await aiofiles.os.path.exists(file_path):
            raise VaultOperationError(f"Note not found: {normalized_path}")

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
                    # Validate content size for the new content being added/replaced
                    test_content = content
                    if append:
                        # For append, check combined size
                        test_content = f"{new_content.rstrip()}\n\n{content}"
                    
                    if not InputValidator.validate_content_size(test_content, self.max_file_size_mb):
                        raise VaultOperationError(
                            f"Content exceeds maximum size of {self.max_file_size_mb}MB"
                        )
                    
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

                # Final validation of complete content size
                if not InputValidator.validate_content_size(new_content, self.max_file_size_mb):
                    raise VaultOperationError(
                        f"Final content exceeds maximum size of {self.max_file_size_mb}MB"
                    )

                # Write updated content atomically
                await self._atomic_write(file_path, new_content)

                logger.info(f"Updated note: {normalized_path}")

                # Read and return the updated note
                return await self.read_note(normalized_path)

            except Exception as e:
                logger.error(f"Error updating note {normalized_path}: {e}")
                raise VaultOperationError(f"Failed to update note: {e}")

    async def delete_note(self, path: str) -> bool:
        """
        Delete a note from the vault with robust error handling.

        Args:
            path: Relative path to the note

        Returns:
            True if deleted successfully

        Raises:
            VaultOperationError: If note cannot be deleted
        """
        file_path, normalized_path = self._validate_path(path)

        async with self._get_lock(path):
            try:
                # Double-check file exists before attempting delete
                if not await aiofiles.os.path.exists(file_path):
                    raise VaultOperationError(f"Note not found: {normalized_path}")
                
                # Additional safety check: ensure it's a file and not a directory
                stat_info = await aiofiles.os.stat(file_path)
                if not stat_info.st_mode & 0o100000:  # S_IFREG check for regular file
                    raise VaultOperationError(f"Path is not a regular file: {normalized_path}")
                
                # Perform the deletion
                await aiofiles.os.remove(file_path)
                
                # Verify deletion was successful
                if await aiofiles.os.path.exists(file_path):
                    raise VaultOperationError(f"File still exists after deletion: {normalized_path}")
                
                logger.info(f"Deleted note: {normalized_path}")
                return True

            except VaultOperationError:
                # Re-raise our own exceptions
                raise
            except FileNotFoundError:
                # File was already deleted, consider it success
                logger.info(f"Note was already deleted: {normalized_path}")
                return True
            except PermissionError as e:
                logger.error(f"Permission denied deleting note {normalized_path}: {e}")
                raise VaultOperationError(f"Permission denied: cannot delete {normalized_path}")
            except Exception as e:
                logger.error(f"Error deleting note {normalized_path}: {e}")
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
