"""
Tests for Obsidian utilities.
"""

import pytest

from mcp_obsidian.obsidian_utils import ObsidianParser


class TestObsidianParser:
    """Tests for ObsidianParser."""

    def test_extract_frontmatter(self):
        """Test frontmatter extraction."""
        content = """---
title: Test Note
tags:
  - test
  - demo
---

# Content here
"""
        frontmatter, content_without = ObsidianParser.extract_frontmatter(content)

        assert frontmatter is not None
        assert frontmatter["title"] == "Test Note"
        assert frontmatter["tags"] == ["test", "demo"]
        assert "# Content here" in content_without

    def test_extract_frontmatter_no_frontmatter(self):
        """Test extraction when no frontmatter exists."""
        content = "# Just content"
        frontmatter, content_without = ObsidianParser.extract_frontmatter(content)

        assert frontmatter is None
        assert content_without == content

    def test_add_frontmatter(self):
        """Test adding frontmatter."""
        content = "# Test Content"
        frontmatter = {"title": "Test", "tags": ["tag1"]}

        result = ObsidianParser.add_frontmatter(content, frontmatter)

        assert result.startswith("---\n")
        assert "title: Test" in result
        assert "# Test Content" in result

    def test_update_frontmatter_merge(self):
        """Test updating frontmatter with merge."""
        content = """---
title: Original
tags:
  - tag1
---

Content
"""
        updates = {"tags": ["tag1", "tag2"], "date": "2024-01-01"}

        result = ObsidianParser.update_frontmatter(content, updates, merge=True)

        assert "tag1" in result
        assert "tag2" in result
        assert "date: '2024-01-01'" in result or "date: 2024-01-01" in result

    def test_extract_wiki_links(self):
        """Test wiki link extraction."""
        content = """
        Link to [[Note 1]] and [[Note 2|Display Text]].
        Also [[Note 3#Heading]].
        """

        links = ObsidianParser.extract_wiki_links(content)

        assert "Note 1" in links
        assert "Note 2" in links
        assert "Note 3" in links
        assert len(links) == 3

    def test_extract_tags(self):
        """Test tag extraction."""
        content = """---
tags:
  - frontmatter-tag
---

Content with #inline-tag and #another-tag.
Also #nested/tag.
"""

        tags = ObsidianParser.extract_tags(content, include_frontmatter=True)

        assert "frontmatter-tag" in tags
        assert "inline-tag" in tags
        assert "another-tag" in tags
        assert "nested/tag" in tags

    def test_extract_tags_no_frontmatter(self):
        """Test tag extraction without frontmatter."""
        content = "Just #inline-tag here."

        tags = ObsidianParser.extract_tags(content, include_frontmatter=False)

        assert "inline-tag" in tags
        assert len(tags) == 1

    def test_add_tag_to_frontmatter(self):
        """Test adding tag to frontmatter."""
        content = """---
tags:
  - existing
---

Content
"""

        result = ObsidianParser.add_tag(content, "newtag", to_frontmatter=True)

        assert "newtag" in result
        assert "existing" in result

    def test_add_tag_inline(self):
        """Test adding inline tag."""
        content = "Some content"

        result = ObsidianParser.add_tag(content, "newtag", to_frontmatter=False)

        assert "#newtag" in result

    def test_remove_tag(self):
        """Test removing tag."""
        content = """---
tags:
  - tag1
  - tag2
---

Content with #tag1 inline.
"""

        result = ObsidianParser.remove_tag(content, "tag1")

        assert "tag1" not in result or result.count("tag1") == 0

    def test_normalize_note_name(self):
        """Test note name normalization."""
        assert ObsidianParser.normalize_note_name("Test Note.md") == "test note"
        assert ObsidianParser.normalize_note_name("folder/Test.md") == "test"
        assert ObsidianParser.normalize_note_name("Test") == "test"

    def test_create_wiki_link(self):
        """Test wiki link creation."""
        link = ObsidianParser.create_wiki_link("Note Name")
        assert link == "[[Note Name]]"

        link_with_display = ObsidianParser.create_wiki_link("Note Name", "Display")
        assert link_with_display == "[[Note Name|Display]]"

    def test_parse_dataview_field(self):
        """Test dataview field parsing."""
        line = "author:: John Doe"
        result = ObsidianParser.parse_dataview_field(line)

        assert result is not None
        assert result[0] == "author"
        assert result[1] == "John Doe"

    def test_parse_dataview_field_invalid(self):
        """Test dataview field parsing with invalid input."""
        line = "Just regular text"
        result = ObsidianParser.parse_dataview_field(line)

        assert result is None

    def test_extract_dataview_fields(self):
        """Test extracting all dataview fields."""
        content = """
        author:: John Doe
        date:: 2024-01-01
        rating:: 5

        Some content here.
        """

        fields = ObsidianParser.extract_dataview_fields(content)

        assert fields["author"] == "John Doe"
        assert fields["date"] == "2024-01-01"
        assert fields["rating"] == "5"
