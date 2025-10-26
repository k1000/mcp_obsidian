# MCP Obsidian Server

A secure, high-performance FastMCP server that provides authenticated access to Obsidian vaults through the Model Context Protocol (MCP).

## Features

- **🔒 Secure Access**: API key authentication with rate limiting
- **✏️ Full CRUD Operations**: Read, create, update, and delete notes
- **🔍 Powerful Search**: Full-text search with tag and frontmatter filtering
- **🏷️ Tag Management**: Extract, add, and remove tags from notes
- **🔗 Link Tracking**: Parse wiki-style links and find backlinks
- **📊 Vault Statistics**: Get insights about your vault
- **⚡ Async Performance**: Built with async I/O for optimal performance
- **✅ Input Validation**: Comprehensive sanitization and security checks
- **📝 Logging & Audit**: Detailed logging with audit trail
- **✔️ Well Tested**: Comprehensive test suite with >80% coverage

## Quick Start

### Installation

```bash
# Clone the repository
cd mcp_obsidian

# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --extra dev
```

### Configuration

1. Copy the example configuration:

```bash
cp config/config.example.yaml config/config.yaml
cp .env.example .env
```

2. Edit `config/config.yaml` and set your vault path:

```yaml
vault:
  path: '/path/to/your/obsidian/vault'

auth:
  api_keys:
    - 'your-secure-api-key-here'
```

3. Or use environment variables in `.env`:

```bash
OBSIDIAN_VAULT_PATH=/path/to/your/vault
OBSIDIAN_API_KEYS=your-api-key-here
```

### Running the Server

```bash
# Run with default config
uv run python -m mcp_obsidian.server

# Or specify a config file
uv run python -m mcp_obsidian.server config/config.yaml
```

## MCP Tools

The server provides the following MCP tools:

### Note Operations

#### `read_note`

Read a note from the vault.

**Parameters:**

- `path` (str): Relative path to the note

**Returns:** JSON with note metadata, content, frontmatter, tags, and links

**Example:**

```python
read_note(path="daily/2024-01-01.md")
```

#### `create_note`

Create a new note in the vault.

**Parameters:**

- `path` (str): Relative path for the new note
- `content` (str): Markdown content
- `frontmatter` (dict, optional): YAML frontmatter data

**Example:**

```python
create_note(
    path="meetings/team-standup.md",
    content="# Team Standup\n\nNotes here...",
    frontmatter={"date": "2024-01-01", "tags": ["meeting"]}
)
```

#### `update_note`

Update an existing note.

**Parameters:**

- `path` (str): Path to the note
- `content` (str, optional): New content
- `frontmatter` (dict, optional): Frontmatter updates
- `append` (bool): If True, append content instead of replacing

**Example:**

```python
update_note(
    path="daily/2024-01-01.md",
    content="Additional notes...",
    append=True
)
```

#### `delete_note`

Delete a note from the vault.

**Parameters:**

- `path` (str): Path to the note

**Example:**

```python
delete_note(path="old-note.md")
```

### Search & Discovery

#### `search_notes`

Search through vault content.

**Parameters:**

- `query` (str): Search query string
- `tags` (list[str], optional): Filter by tags
- `path_pattern` (str, optional): Filter by path pattern
- `limit` (int): Maximum results (default: 100)
- `offset` (int): Offset for pagination
- `case_sensitive` (bool): Case-sensitive search

**Example:**

```python
search_notes(
    query="project planning",
    tags=["work", "important"],
    limit=50
)
```

#### `list_notes`

List notes in the vault.

**Parameters:**

- `path_pattern` (str, optional): Glob pattern for filtering
- `limit` (int): Maximum results
- `offset` (int): Pagination offset
- `sort_by` (str): Sort field (name, modified, created, size)
- `sort_desc` (bool): Sort descending

**Example:**

```python
list_notes(
    path_pattern="projects/**/*.md",
    sort_by="modified",
    limit=100
)
```

#### `get_backlinks`

Find all notes that link to a specific note.

**Parameters:**

- `note_name` (str): Name of the note

**Example:**

```python
get_backlinks(note_name="important-concept")
```

### Tag Management

#### `add_tag_to_note`

Add a tag to a note.

**Parameters:**

- `path` (str): Path to the note
- `tag` (str): Tag to add (without # prefix)
- `to_frontmatter` (bool): Add to frontmatter vs inline

**Example:**

```python
add_tag_to_note(
    path="note.md",
    tag="reviewed",
    to_frontmatter=True
)
```

#### `remove_tag_from_note`

Remove a tag from a note.

**Parameters:**

- `path` (str): Path to the note
- `tag` (str): Tag to remove

**Example:**

```python
remove_tag_from_note(path="note.md", tag="draft")
```

### Vault Information

#### `get_vault_stats`

Get statistics about the vault.

**Returns:** JSON with total notes, size, tags, and more

**Example:**

```python
get_vault_stats()
```

## Configuration Reference

### Server Configuration

```yaml
server:
  host: '0.0.0.0' # Server host
  port: 8000 # Server port
  log_level: 'INFO' # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Vault Configuration

```yaml
vault:
  path: '/path/to/vault' # Absolute path to Obsidian vault
  note_extension: 'md' # Note file extension
  max_file_size_mb: 10 # Maximum file size
  allowed_extensions: # Allowed file types
    - 'md'
    - 'pdf'
    - 'png'
```

### Authentication

```yaml
auth:
  enabled: true
  api_keys: # List of valid API keys
    - 'your-key-here'
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
```

### Search Configuration

```yaml
search:
  enabled: true
  max_results: 100
  cache_enabled: true
  cache_ttl_seconds: 300
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_vault.py
```

### Code Quality

```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run mypy src
```

## Security Best Practices

1. **API Keys**: Use strong, randomly generated API keys (minimum 16 characters)
2. **Environment Variables**: Store sensitive data in `.env` file, never commit to git
3. **Rate Limiting**: Enable rate limiting to prevent abuse
4. **Path Validation**: All paths are validated to prevent directory traversal
5. **Input Sanitization**: All inputs are sanitized before processing
6. **Audit Logging**: Enable audit logging to track all operations

## Architecture

```
src/mcp_obsidian/
├── __init__.py           # Package initialization
├── models.py             # Pydantic data models
├── config.py             # Configuration management
├── auth.py               # Authentication & security
├── vault.py              # Core vault operations
├── obsidian_utils.py     # Obsidian-specific utilities
├── search.py             # Search functionality
├── logging_config.py     # Logging configuration
└── server.py             # MCP server implementation
```

## Troubleshooting

### Common Issues

**Issue: "Vault path does not exist"**

- Ensure the path in `config.yaml` is correct and absolute
- Check file permissions

**Issue: "Invalid API key"**

- Verify API key in config matches the one being used
- Check that authentication is enabled

**Issue: "Rate limit exceeded"**

- Wait for rate limit window to reset
- Adjust rate limit settings in config

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:

- Open an issue on GitHub
- Check existing documentation
- Review configuration examples

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Designed for [Obsidian](https://obsidian.md)
- Uses [Pydantic](https://docs.pydantic.dev) for validation
