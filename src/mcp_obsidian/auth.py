"""
Authentication and security utilities.
"""

import hashlib
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PathValidator:
    """Validates and sanitizes file paths to prevent directory traversal attacks."""

    @staticmethod
    def is_safe_path(vault_root: Path, target_path: str) -> bool:
        """
        Check if a path is safe (no directory traversal).

        Args:
            vault_root: Root directory of the vault
            target_path: Path to validate (relative to vault root)

        Returns:
            True if path is safe, False otherwise
        """
        # Prevent obvious attacks
        if ".." in target_path or target_path.startswith("/"):
            return False

        # Resolve to absolute path and check it's within vault
        try:
            full_path = (vault_root / target_path).resolve()
            vault_root_resolved = vault_root.resolve()

            # Check if the path is within the vault root
            return str(full_path).startswith(str(vault_root_resolved))
        except (ValueError, OSError):
            return False

    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize a path by removing dangerous components.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path
        """
        # Remove leading/trailing slashes and spaces
        path = path.strip().strip("/")

        # Remove any .. components
        parts = path.split("/")
        safe_parts = [p for p in parts if p and p != ".."]

        return "/".join(safe_parts)


class ApiKeyAuth:
    """Simple API key authentication."""

    def __init__(self, api_keys: List[str]):
        """
        Initialize API key authentication.

        Args:
            api_keys: List of valid API keys
        """
        # Store hashed keys for security
        self.valid_keys_hashed = {self._hash_key(key) for key in api_keys}
        logger.info(f"Initialized API key auth with {len(api_keys)} keys")

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def verify_key(self, api_key: str) -> bool:
        """
        Verify an API key.

        Args:
            api_key: API key to verify

        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False

        hashed = self._hash_key(api_key)
        is_valid = hashed in self.valid_keys_hashed

        if is_valid:
            logger.debug("API key verified successfully")
        else:
            logger.warning("Invalid API key attempt")

        return is_valid

    def extract_from_header(self, authorization_header: Optional[str]) -> Optional[str]:
        """
        Extract API key from Authorization header.

        Args:
            authorization_header: Authorization header value

        Returns:
            API key if found, None otherwise
        """
        if not authorization_header:
            return None

        # Support both "Bearer <key>" and just "<key>"
        parts = authorization_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        elif len(parts) == 1:
            return parts[0]

        return None


class TokenBucket:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_tokens(self) -> float:
        """Get current token count."""
        self._refill()
        return self.tokens


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size

        # Store buckets per client (identified by API key or IP)
        self.buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=burst_size, refill_rate=requests_per_minute / 60.0
            )
        )

        logger.info(
            f"Rate limiter initialized: {requests_per_minute} req/min, burst={burst_size}"
        )

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if a client can make a request.

        Args:
            client_id: Unique identifier for the client

        Returns:
            True if request is allowed, False if rate limited
        """
        bucket = self.buckets[client_id]
        allowed = bucket.consume(1)

        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id[:8]}...")

        return allowed

    def get_remaining(self, client_id: str) -> float:
        """
        Get remaining tokens for a client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Number of remaining tokens
        """
        bucket = self.buckets[client_id]
        return bucket.get_tokens()


class InputValidator:
    """Validates and sanitizes user inputs."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename to remove dangerous characters.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path separators and other dangerous chars
        dangerous_chars = ["\\", "/", ":", "*", "?", '"', "<", ">", "|", "\0"]
        for char in dangerous_chars:
            filename = filename.replace(char, "_")

        # Remove leading/trailing dots and spaces
        filename = filename.strip(". ")

        # Ensure it's not empty
        if not filename:
            filename = "untitled"

        return filename

    @staticmethod
    def sanitize_note_path(path: str) -> str:
        """
        Sanitize a note path while preserving folder structure.

        Args:
            path: Relative path that may include directories and filename

        Returns:
            Sanitized relative path
        """
        if not path:
            return "untitled.md"

        normalized = path.strip().lstrip("/")
        if not normalized:
            return "untitled.md"

        parts = [part for part in normalized.split("/") if part]
        if not parts:
            return "untitled.md"

        *dirs, filename = parts

        safe_dirs = []
        for part in dirs:
            stripped = part.strip()
            if not stripped or stripped in {".", ".."}:
                continue
            safe_dirs.append(stripped)

        safe_filename = InputValidator.sanitize_filename(filename)

        safe_parts = safe_dirs + [safe_filename]
        return "/".join(safe_parts)

    @staticmethod
    def validate_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """
        Validate file extension.

        Args:
            filename: Filename to check
            allowed_extensions: List of allowed extensions (without dots)

        Returns:
            True if extension is allowed, False otherwise
        """
        if "." not in filename:
            return False

        extension = filename.rsplit(".", 1)[1].lower()
        return extension in [ext.lower().lstrip(".") for ext in allowed_extensions]

    @staticmethod
    def validate_content_size(content: str, max_size_mb: int = 10) -> bool:
        """
        Validate content size.

        Args:
            content: Content to validate
            max_size_mb: Maximum size in MB

        Returns:
            True if size is acceptable, False otherwise
        """
        size_bytes = len(content.encode("utf-8"))
        max_bytes = max_size_mb * 1024 * 1024

        if size_bytes > max_bytes:
            logger.warning(
                f"Content size {size_bytes} exceeds limit {max_bytes}"
            )
            return False

        return True
