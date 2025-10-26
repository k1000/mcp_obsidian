"""
Configuration management for MCP Obsidian Server.
Supports YAML config files and environment variables.
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v


class VaultConfig(BaseSettings):
    """Vault configuration."""

    path: str = Field(..., description="Path to Obsidian vault")
    note_extension: str = Field(default="md", description="Note file extension")
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Max file size in MB")
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ["md", "pdf", "png", "jpg", "jpeg", "gif", "svg"],
        description="Allowed file extensions",
    )

    @field_validator("path")
    @classmethod
    def validate_vault_path(cls, v: str) -> str:
        """Validate vault path exists."""
        path = Path(v).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Vault path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Vault path is not a directory: {v}")
        return str(path)


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(
        default=60, ge=1, le=1000, description="Requests per minute"
    )
    burst_size: int = Field(default=10, ge=1, le=100, description="Burst size")


class AuthConfig(BaseSettings):
    """Authentication configuration."""

    enabled: bool = Field(default=True, description="Enable authentication")
    api_keys: List[str] = Field(
        default_factory=list, description="List of valid API keys"
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting config"
    )

    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, v: List[str]) -> List[str]:
        """Validate API keys."""
        if not v:
            raise ValueError("At least one API key must be configured when auth is enabled")
        for key in v:
            if len(key) < 16:
                raise ValueError("API keys must be at least 16 characters long")
        return v


class CorsConfig(BaseSettings):
    """CORS configuration."""

    enabled: bool = Field(default=True, description="Enable CORS")
    allow_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed origins",
    )
    allow_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE"],
        description="Allowed HTTP methods",
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type"],
        description="Allowed headers",
    )


class SearchConfig(BaseSettings):
    """Search configuration."""

    enabled: bool = Field(default=True, description="Enable search")
    max_results: int = Field(
        default=100, ge=1, le=1000, description="Maximum search results"
    )
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(
        default=300, ge=60, le=3600, description="Cache TTL in seconds"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    file: Optional[str] = Field(default=None, description="Log file path")
    format: str = Field(default="json", description="Log format: json or text")
    log_requests: bool = Field(default=True, description="Log requests/responses")
    audit_enabled: bool = Field(default=True, description="Enable audit logging")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate log format."""
        if v not in ["json", "text"]:
            raise ValueError("Log format must be 'json' or 'text'")
        return v


class OllamaProviderConfig(BaseSettings):
    """Ollama embedding provider configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")


class OpenAIProviderConfig(BaseSettings):
    """OpenAI embedding provider configuration."""

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    dimensions: Optional[int] = Field(
        default=None, description="Embedding dimensions (for models that support it)"
    )


class HuggingFaceProviderConfig(BaseSettings):
    """HuggingFace embedding provider configuration."""

    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name",
    )
    device: str = Field(default="cpu", description="Device: cpu or cuda")
    use_api: bool = Field(default=False, description="Use HuggingFace API instead of local")
    api_key: str = Field(default="", description="HuggingFace API key (if use_api=True)")


class CohereProviderConfig(BaseSettings):
    """Cohere embedding provider configuration."""

    api_key: str = Field(default="", description="Cohere API key")
    model: str = Field(default="embed-english-v3.0", description="Cohere embedding model")


class RAGProvidersConfig(BaseSettings):
    """Configuration for all embedding providers."""

    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    huggingface: HuggingFaceProviderConfig = Field(default_factory=HuggingFaceProviderConfig)
    cohere: CohereProviderConfig = Field(default_factory=CohereProviderConfig)


class RAGChunkingConfig(BaseSettings):
    """Document chunking configuration."""

    strategy: str = Field(default="smart", description="Chunking strategy: smart, fixed, recursive")
    chunk_size: int = Field(default=512, ge=128, le=2048, description="Chunk size in tokens")
    chunk_overlap: int = Field(
        default=50, ge=0, le=512, description="Overlap between chunks in tokens"
    )
    split_on_headers: bool = Field(
        default=True, description="Split on markdown headers for smart chunking"
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate chunking strategy."""
        if v not in ["smart", "fixed", "recursive"]:
            raise ValueError("Chunking strategy must be 'smart', 'fixed', or 'recursive'")
        return v


class RAGSearchConfig(BaseSettings):
    """RAG search configuration."""

    hybrid_mode: bool = Field(
        default=True, description="Combine semantic and keyword search"
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic results in hybrid mode",
    )
    keyword_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for keyword results in hybrid mode"
    )
    top_k: int = Field(default=20, ge=1, le=100, description="Number of results to retrieve")
    rerank: bool = Field(default=False, description="Enable re-ranking of results")


class RAGConfig(BaseSettings):
    """RAG and semantic search configuration."""

    enabled: bool = Field(default=False, description="Enable RAG/semantic search")
    provider: str = Field(
        default="ollama",
        description="Embedding provider: ollama, openai, huggingface, cohere",
    )
    providers: RAGProvidersConfig = Field(default_factory=RAGProvidersConfig)
    vector_store: str = Field(default="chroma", description="Vector store: chroma or faiss")
    vector_db_path: str = Field(
        default="data/vector_db", description="Path to vector database"
    )
    chunking: RAGChunkingConfig = Field(default_factory=RAGChunkingConfig)
    search: RAGSearchConfig = Field(default_factory=RAGSearchConfig)
    cache_embeddings: bool = Field(default=True, description="Cache embeddings")
    batch_size: int = Field(
        default=32, ge=1, le=100, description="Batch size for embedding generation"
    )
    index_on_startup: bool = Field(
        default=False, description="Automatically index vault on startup"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate embedding provider."""
        valid_providers = ["ollama", "openai", "huggingface", "cohere"]
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @field_validator("vector_store")
    @classmethod
    def validate_vector_store(cls, v: str) -> str:
        """Validate vector store."""
        if v not in ["chroma", "faiss"]:
            raise ValueError("Vector store must be 'chroma' or 'faiss'")
        return v


class Config(BaseSettings):
    """Main configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OBSIDIAN_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    vault: VaultConfig
    auth: AuthConfig = Field(default_factory=AuthConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from file and environment variables.
        Environment variables take precedence over file values.
        """
        # Default config path
        if config_path is None:
            config_path = os.getenv("OBSIDIAN_CONFIG_PATH", "config/config.yaml")

        # Load from YAML if exists
        if Path(config_path).exists():
            return cls.from_yaml(config_path)

        # Otherwise load from environment variables only
        return cls()


def get_config(config_path: Optional[str] = None) -> Config:
    """Get application configuration."""
    return Config.load(config_path)
