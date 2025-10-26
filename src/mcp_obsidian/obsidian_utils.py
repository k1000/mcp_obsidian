"""
Obsidian-specific utilities for parsing frontmatter, wiki links, tags, etc.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ObsidianParser:
    """Parser for Obsidian-specific markdown features."""

    # Regex patterns
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
    TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z0-9_/-]+)")
    INLINE_TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z0-9_/-]+)")

    @staticmethod
    def extract_frontmatter(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Extract YAML frontmatter from content.

        Args:
            content: The full note content

        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)
        """
        match = ObsidianParser.FRONTMATTER_PATTERN.match(content)
        if not match:
            return None, content

        frontmatter_text = match.group(1)
        content_without = content[match.end() :]

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
            # Ensure it's a dict
            if not isinstance(frontmatter, dict):
                return None, content
            return frontmatter, content_without
        except yaml.YAMLError:
            return None, content

    @staticmethod
    def add_frontmatter(content: str, frontmatter: Dict[str, Any]) -> str:
        """
        Add or replace frontmatter in content.

        Args:
            content: The note content
            frontmatter: Dictionary of frontmatter data

        Returns:
            Content with frontmatter
        """
        # Remove existing frontmatter if present
        _, content_without = ObsidianParser.extract_frontmatter(content)

        # Convert frontmatter to YAML
        frontmatter_yaml = yaml.dump(
            frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        return f"---\n{frontmatter_yaml}---\n{content_without}"

    @staticmethod
    def update_frontmatter(
        content: str, updates: Dict[str, Any], merge: bool = True
    ) -> str:
        """
        Update frontmatter fields.

        Args:
            content: The note content
            updates: Dictionary of fields to update
            merge: If True, merge with existing frontmatter; if False, replace

        Returns:
            Content with updated frontmatter
        """
        existing_fm, content_without = ObsidianParser.extract_frontmatter(content)

        if merge and existing_fm:
            # Merge updates into existing frontmatter
            existing_fm.update(updates)
            frontmatter = existing_fm
        else:
            frontmatter = updates

        return ObsidianParser.add_frontmatter(content_without, frontmatter)

    @staticmethod
    def extract_wiki_links(content: str) -> List[str]:
        """
        Extract all wiki-style links from content.

        Args:
            content: The note content

        Returns:
            List of linked note names (without brackets)
        """
        matches = ObsidianParser.WIKI_LINK_PATTERN.findall(content)
        links = []

        for match in matches:
            # Handle [[Note Name|Display Text]] format
            if "|" in match:
                link_target = match.split("|")[0].strip()
            else:
                link_target = match.strip()

            # Handle [[Note Name#Heading]] format
            if "#" in link_target:
                link_target = link_target.split("#")[0].strip()

            if link_target:
                links.append(link_target)

        return links

    @staticmethod
    def extract_tags(content: str, include_frontmatter: bool = True) -> List[str]:
        """
        Extract all tags from content.

        Args:
            content: The note content
            include_frontmatter: Whether to include tags from frontmatter

        Returns:
            List of unique tags (without # prefix)
        """
        tags = set()

        # Extract tags from frontmatter
        if include_frontmatter:
            frontmatter, _ = ObsidianParser.extract_frontmatter(content)
            if frontmatter:
                fm_tags = frontmatter.get("tags", [])
                if isinstance(fm_tags, list):
                    tags.update(str(tag).strip() for tag in fm_tags)
                elif isinstance(fm_tags, str):
                    tags.update(tag.strip() for tag in fm_tags.split(","))

        # Extract inline tags from content
        inline_tags = ObsidianParser.TAG_PATTERN.findall(content)
        tags.update(inline_tags)

        return sorted(list(tags))

    @staticmethod
    def add_tag(content: str, tag: str, to_frontmatter: bool = True) -> str:
        """
        Add a tag to the note.

        Args:
            content: The note content
            tag: Tag to add (without # prefix)
            to_frontmatter: If True, add to frontmatter; otherwise append inline

        Returns:
            Content with added tag
        """
        tag = tag.lstrip("#").strip()

        if to_frontmatter:
            frontmatter, content_without = ObsidianParser.extract_frontmatter(content)
            if frontmatter is None:
                frontmatter = {}

            # Add tag to frontmatter tags list
            fm_tags = frontmatter.get("tags", [])
            if not isinstance(fm_tags, list):
                fm_tags = []

            if tag not in fm_tags:
                fm_tags.append(tag)
                frontmatter["tags"] = fm_tags

            return ObsidianParser.add_frontmatter(content_without, frontmatter)
        else:
            # Add inline tag at the end
            return f"{content.rstrip()}\n\n#{tag}"

    @staticmethod
    def remove_tag(content: str, tag: str) -> str:
        """
        Remove a tag from the note (both frontmatter and inline).

        Args:
            content: The note content
            tag: Tag to remove (without # prefix)

        Returns:
            Content with tag removed
        """
        tag = tag.lstrip("#").strip()

        # Remove from frontmatter
        frontmatter, content_without = ObsidianParser.extract_frontmatter(content)
        if frontmatter:
            fm_tags = frontmatter.get("tags", [])
            if isinstance(fm_tags, list) and tag in fm_tags:
                fm_tags.remove(tag)
                frontmatter["tags"] = fm_tags
                content_without = ObsidianParser.add_frontmatter(
                    content_without, frontmatter
                ).split("---\n", 2)[2]

        # Remove inline tags
        content_without = re.sub(rf"(?:^|\s)#{re.escape(tag)}(?:\s|$)", " ", content_without)

        return content_without.strip()

    @staticmethod
    def normalize_note_name(name: str) -> str:
        """
        Normalize note name for comparison.

        Args:
            name: Note name or path

        Returns:
            Normalized name (lowercase, no extension, no path)
        """
        # Remove .md extension
        if name.endswith(".md"):
            name = name[:-3]

        # Get just the filename
        if "/" in name:
            name = name.split("/")[-1]

        return name.lower().strip()

    @staticmethod
    def create_wiki_link(note_name: str, display_text: Optional[str] = None) -> str:
        """
        Create a wiki-style link.

        Args:
            note_name: Name of the note to link to
            display_text: Optional display text

        Returns:
            Wiki link string
        """
        if display_text:
            return f"[[{note_name}|{display_text}]]"
        return f"[[{note_name}]]"

    @staticmethod
    def parse_dataview_field(line: str) -> Optional[Tuple[str, str]]:
        """
        Parse dataview inline field (e.g., "key:: value").

        Args:
            line: Line of text to parse

        Returns:
            Tuple of (key, value) or None if not a dataview field
        """
        match = re.match(r"^([^:]+)::\s*(.*)$", line.strip())
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None

    @staticmethod
    def extract_dataview_fields(content: str) -> Dict[str, str]:
        """
        Extract all dataview inline fields from content.

        Args:
            content: The note content

        Returns:
            Dictionary of dataview fields
        """
        fields = {}
        for line in content.split("\n"):
            result = ObsidianParser.parse_dataview_field(line)
            if result:
                key, value = result
                fields[key] = value
        return fields
