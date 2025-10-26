"""
Tests for vault operations.
"""

import pytest

from mcp_obsidian.vault import VaultManager, VaultOperationError


class TestVaultManager:
    """Tests for VaultManager."""

    @pytest.mark.asyncio
    async def test_read_note(self, vault_manager: VaultManager):
        """Test reading a note."""
        note = await vault_manager.read_note("test1.md")

        assert note.metadata.name == "test1.md"
        assert "Test Note 1" in note.content
        assert "test" in note.tags
        assert "demo" in note.tags
        assert "test2" in note.links

    @pytest.mark.asyncio
    async def test_read_note_not_found(self, vault_manager: VaultManager):
        """Test reading non-existent note."""
        with pytest.raises(VaultOperationError, match="Note not found"):
            await vault_manager.read_note("nonexistent.md")

    @pytest.mark.asyncio
    async def test_create_note(self, vault_manager: VaultManager):
        """Test creating a new note."""
        content = "# New Note\n\nThis is new content."
        frontmatter = {"title": "New Note", "tags": ["new"]}

        note = await vault_manager.create_note("new.md", content, frontmatter)

        assert note.metadata.name == "new.md"
        assert "New Note" in note.content
        assert note.frontmatter is not None
        assert note.frontmatter.title == "New Note"

    @pytest.mark.asyncio
    async def test_create_note_already_exists(self, vault_manager: VaultManager):
        """Test creating note that already exists."""
        with pytest.raises(VaultOperationError, match="already exists"):
            await vault_manager.create_note(
                "test1.md", "Content", None
            )

    @pytest.mark.asyncio
    async def test_update_note_content(self, vault_manager: VaultManager):
        """Test updating note content."""
        new_content = "# Updated Content"
        note = await vault_manager.update_note("test1.md", content=new_content)

        assert "Updated Content" in note.content

    @pytest.mark.asyncio
    async def test_update_note_append(self, vault_manager: VaultManager):
        """Test appending to note."""
        original = await vault_manager.read_note("test1.md")
        append_text = "\n\nAppended text."

        updated = await vault_manager.update_note(
            "test1.md", content=append_text, append=True
        )

        assert "Appended text" in updated.content
        assert len(updated.content) > len(original.content)

    @pytest.mark.asyncio
    async def test_update_note_frontmatter(self, vault_manager: VaultManager):
        """Test updating note frontmatter."""
        updates = {"tags": ["test", "demo", "updated"]}

        note = await vault_manager.update_note("test1.md", frontmatter=updates)

        assert "updated" in note.tags

    @pytest.mark.asyncio
    async def test_delete_note(self, vault_manager: VaultManager):
        """Test deleting a note."""
        # Create a note to delete
        await vault_manager.create_note("to_delete.md", "Content")

        # Delete it
        result = await vault_manager.delete_note("to_delete.md")
        assert result is True

        # Verify it's gone
        with pytest.raises(VaultOperationError, match="Note not found"):
            await vault_manager.read_note("to_delete.md")

    @pytest.mark.asyncio
    async def test_delete_note_not_found(self, vault_manager: VaultManager):
        """Test deleting non-existent note."""
        with pytest.raises(VaultOperationError, match="Note not found"):
            await vault_manager.delete_note("nonexistent.md")

    @pytest.mark.asyncio
    async def test_list_notes(self, vault_manager: VaultManager):
        """Test listing notes."""
        notes = await vault_manager.list_notes()

        assert len(notes) >= 3  # test1, test2, nested
        names = [n.name for n in notes]
        assert "test1.md" in names
        assert "test2.md" in names

    @pytest.mark.asyncio
    async def test_list_notes_pattern(self, vault_manager: VaultManager):
        """Test listing notes with pattern."""
        notes = await vault_manager.list_notes(path_pattern="folder/*.md")

        assert len(notes) == 1
        assert notes[0].name == "nested.md"

    @pytest.mark.asyncio
    async def test_get_vault_stats(self, vault_manager: VaultManager):
        """Test getting vault statistics."""
        stats = await vault_manager.get_vault_stats()

        assert stats.total_notes >= 3
        assert stats.total_size_bytes > 0
        assert len(stats.unique_tags) > 0
        assert "md" in stats.note_count_by_extension
