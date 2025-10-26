"""
Search functionality for Obsidian vault.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import SearchQuery, SearchResponse, SearchResult
from .obsidian_utils import ObsidianParser
from .vault import VaultManager

logger = logging.getLogger(__name__)


class SearchEngine:
    """Search engine for Obsidian vault."""

    def __init__(self, vault_manager: VaultManager, cache_enabled: bool = True):
        """
        Initialize search engine.

        Args:
            vault_manager: VaultManager instance
            cache_enabled: Whether to enable result caching
        """
        self.vault_manager = vault_manager
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Tuple[SearchResponse, float]] = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info("Initialized SearchEngine")

    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key from query."""
        return f"{query.query}:{query.tags}:{query.path_pattern}:{query.frontmatter_filter}"

    def _get_cached_result(self, query: SearchQuery) -> Optional[SearchResponse]:
        """Get cached search result if valid."""
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(query)
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug("Returning cached search result")
                return result
            else:
                # Remove expired cache entry
                del self._cache[cache_key]

        return None

    def _cache_result(self, query: SearchQuery, result: SearchResponse) -> None:
        """Cache search result."""
        if self.cache_enabled:
            cache_key = self._get_cache_key(query)
            self._cache[cache_key] = (result, time.time())

    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Search the vault based on query parameters.

        Args:
            query: SearchQuery object with search parameters

        Returns:
            SearchResponse with results
        """
        start_time = time.time()

        # Check cache
        cached = self._get_cached_result(query)
        if cached:
            return cached

        # Get all notes
        notes = await self.vault_manager.list_notes(path_pattern=query.path_pattern)

        results: List[SearchResult] = []

        for note_metadata in notes:
            try:
                # Read note content
                note = await self.vault_manager.read_note(note_metadata.path)

                # Apply filters
                if not self._matches_filters(note, query):
                    continue

                # Search in content
                matches, score = self._search_content(note.content, query.query, query.case_sensitive)

                if score > 0:
                    results.append(
                        SearchResult(
                            note=note_metadata,
                            score=score,
                            matches=matches,
                            match_count=len(matches),
                        )
                    )

            except Exception as e:
                logger.warning(f"Error searching note {note_metadata.path}: {e}")
                continue

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply pagination
        total = len(results)
        results = results[query.offset : query.offset + query.limit]

        duration_ms = (time.time() - start_time) * 1000

        response = SearchResponse(
            results=results,
            total=total,
            query=query,
            duration_ms=duration_ms,
        )

        # Cache result
        self._cache_result(query, response)

        logger.info(
            f"Search completed: {total} results in {duration_ms:.2f}ms (query: '{query.query}')"
        )

        return response

    def _matches_filters(self, note, query: SearchQuery) -> bool:
        """Check if note matches query filters."""
        # Tag filter
        if query.tags:
            if not any(tag in note.tags for tag in query.tags):
                return False

        # Frontmatter filter
        if query.frontmatter_filter and note.frontmatter:
            for key, value in query.frontmatter_filter.items():
                if not hasattr(note.frontmatter, key):
                    return False
                if getattr(note.frontmatter, key) != value:
                    return False

        return True

    def _search_content(
        self, content: str, query: str, case_sensitive: bool = False
    ) -> Tuple[List[str], float]:
        """
        Search for query in content and return matches with context.

        Args:
            content: Content to search
            query: Search query
            case_sensitive: Whether search is case-sensitive

        Returns:
            Tuple of (matches_with_context, score)
        """
        if not query:
            return [], 0.0

        matches = []
        score = 0.0

        # Prepare query for regex
        if not case_sensitive:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            search_content = content
        else:
            pattern = re.compile(re.escape(query))
            search_content = content

        # Find all matches
        for match in pattern.finditer(search_content):
            start = match.start()
            end = match.end()

            # Extract context (50 chars before and after)
            context_start = max(0, start - 50)
            context_end = min(len(content), end + 50)

            context = content[context_start:context_end]

            # Clean up context
            context = context.replace("\n", " ").strip()

            # Add ellipsis if truncated
            if context_start > 0:
                context = "..." + context
            if context_end < len(content):
                context = context + "..."

            matches.append(context)

            # Increase score
            score += 1.0

        # Boost score for matches in title or frontmatter
        frontmatter, _ = ObsidianParser.extract_frontmatter(content)
        if frontmatter:
            title = frontmatter.get("title", "")
            if query.lower() in str(title).lower():
                score += 5.0

        # Normalize score
        score = min(score, 100.0)

        return matches[:10], score  # Return max 10 matches

    async def search_by_tags(self, tags: List[str]) -> List[SearchResult]:
        """
        Search notes by tags.

        Args:
            tags: List of tags to search for

        Returns:
            List of search results
        """
        query = SearchQuery(query="", tags=tags, limit=1000)
        response = await self.search(query)
        return response.results

    async def search_by_links(self, note_name: str) -> List[str]:
        """
        Find all notes that link to a specific note (backlinks).

        Args:
            note_name: Name of the note to find backlinks for

        Returns:
            List of note paths that link to this note
        """
        normalized_name = ObsidianParser.normalize_note_name(note_name)
        backlinks = []

        notes = await self.vault_manager.list_notes()

        for note_metadata in notes:
            try:
                note = await self.vault_manager.read_note(note_metadata.path)

                # Check if this note contains a link to the target note
                for link in note.links:
                    if ObsidianParser.normalize_note_name(link) == normalized_name:
                        backlinks.append(note_metadata.path)
                        break

            except Exception as e:
                logger.warning(f"Error checking backlinks in {note_metadata.path}: {e}")
                continue

        return backlinks

    def clear_cache(self) -> int:
        """
        Clear search cache.

        Returns:
            Number of cache entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count
