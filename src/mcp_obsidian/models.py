"""
Data models for MCP Obsidian Server using Pydantic v2.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class NoteMetadata(BaseModel):
    """Metadata for a note file."""

    path: str = Field(..., description="Relative path to the note within the vault")
    name: str = Field(..., description="Name of the note file")
    size: int = Field(..., description="File size in bytes")
    created: datetime = Field(..., description="Creation timestamp")
    modified: datetime = Field(..., description="Last modification timestamp")
    extension: str = Field(default="md", description="File extension")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path doesn't contain dangerous patterns."""
        if ".." in v or v.startswith("/"):
            raise ValueError("Path must be relative and not contain '..'")
        return v


class Frontmatter(BaseModel):
    """YAML frontmatter data."""

    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    date: Optional[datetime] = None
    aliases: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # Allow additional fields in frontmatter


class Note(BaseModel):
    """Complete note representation."""

    metadata: NoteMetadata
    content: str = Field(..., description="Full content of the note")
    frontmatter: Optional[Frontmatter] = Field(
        default=None, description="Parsed YAML frontmatter"
    )
    tags: List[str] = Field(default_factory=list, description="Tags extracted from content")
    links: List[str] = Field(
        default_factory=list, description="Wiki-style links [[link]] found in content"
    )
    backlinks: List[str] = Field(
        default_factory=list, description="Notes that link to this note"
    )


class CreateNoteRequest(BaseModel):
    """Request to create a new note."""

    path: str = Field(..., description="Relative path for the new note")
    content: str = Field(..., description="Content of the note")
    frontmatter: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional frontmatter data"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate and sanitize path."""
        if ".." in v or v.startswith("/"):
            raise ValueError("Path must be relative and not contain '..'")
        if not v.endswith(".md"):
            v = f"{v}.md"
        return v


class UpdateNoteRequest(BaseModel):
    """Request to update an existing note."""

    content: Optional[str] = Field(default=None, description="New content")
    frontmatter: Optional[Dict[str, Any]] = Field(default=None, description="New frontmatter")
    append: bool = Field(default=False, description="Append to existing content")


class SearchQuery(BaseModel):
    """Search query parameters."""

    query: str = Field(..., description="Search query string")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    path_pattern: Optional[str] = Field(
        default=None, description="Filter by path pattern (glob)"
    )
    frontmatter_filter: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter by frontmatter fields"
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    case_sensitive: bool = Field(default=False, description="Case-sensitive search")


class SearchResult(BaseModel):
    """Single search result."""

    note: NoteMetadata
    score: float = Field(..., description="Relevance score")
    matches: List[str] = Field(
        default_factory=list, description="Matched text snippets with context"
    )
    match_count: int = Field(..., description="Number of matches found")


class SearchResponse(BaseModel):
    """Search results response."""

    results: List[SearchResult]
    total: int = Field(..., description="Total number of results")
    query: SearchQuery
    duration_ms: float = Field(..., description="Search duration in milliseconds")


class ListNotesRequest(BaseModel):
    """Request to list notes."""

    path_pattern: Optional[str] = Field(default=None, description="Filter by path pattern")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    sort_by: str = Field(default="modified", description="Sort by: name, modified, created, size")
    sort_desc: bool = Field(default=True, description="Sort in descending order")


class OperationStatus(str, Enum):
    """Status of an operation."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class OperationResponse(BaseModel):
    """Generic operation response."""

    status: OperationStatus
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None


class VaultStats(BaseModel):
    """Statistics about the vault."""

    total_notes: int
    total_size_bytes: int
    total_tags: int
    unique_tags: List[str]
    last_modified: datetime
    note_count_by_extension: Dict[str, int]
