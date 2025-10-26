"""
Tests for authentication and security.
"""

import time
from pathlib import Path

import pytest

from mcp_obsidian.auth import (
    ApiKeyAuth,
    InputValidator,
    PathValidator,
    RateLimiter,
    TokenBucket,
)


class TestPathValidator:
    """Tests for PathValidator."""

    def test_is_safe_path_valid(self, tmp_path: Path):
        """Test safe path validation."""
        assert PathValidator.is_safe_path(tmp_path, "test.md")
        assert PathValidator.is_safe_path(tmp_path, "folder/test.md")

    def test_is_safe_path_invalid(self, tmp_path: Path):
        """Test unsafe path detection."""
        assert not PathValidator.is_safe_path(tmp_path, "../outside.md")
        assert not PathValidator.is_safe_path(tmp_path, "/absolute/path.md")
        assert not PathValidator.is_safe_path(tmp_path, "folder/../../outside.md")

    def test_sanitize_path(self):
        """Test path sanitization."""
        assert PathValidator.sanitize_path("/absolute/path") == "absolute/path"
        assert PathValidator.sanitize_path("folder/../file.md") == "folder/file.md"
        assert PathValidator.sanitize_path("  /path/  ") == "path"


class TestApiKeyAuth:
    """Tests for API key authentication."""

    def test_verify_valid_key(self):
        """Test valid API key verification."""
        auth = ApiKeyAuth(["test-api-key-12345678"])
        assert auth.verify_key("test-api-key-12345678")

    def test_verify_invalid_key(self):
        """Test invalid API key rejection."""
        auth = ApiKeyAuth(["test-api-key-12345678"])
        assert not auth.verify_key("wrong-key")
        assert not auth.verify_key("")

    def test_extract_from_bearer_header(self):
        """Test extracting key from Bearer header."""
        auth = ApiKeyAuth(["test-key"])
        key = auth.extract_from_header("Bearer test-key")
        assert key == "test-key"

    def test_extract_from_simple_header(self):
        """Test extracting key from simple header."""
        auth = ApiKeyAuth(["test-key"])
        key = auth.extract_from_header("test-key")
        assert key == "test-key"

    def test_extract_from_invalid_header(self):
        """Test extraction from invalid header."""
        auth = ApiKeyAuth(["test-key"])
        assert auth.extract_from_header("") is None
        assert auth.extract_from_header(None) is None


class TestTokenBucket:
    """Tests for TokenBucket rate limiting."""

    def test_initial_capacity(self):
        """Test initial token capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.get_tokens() == 10.0

    def test_consume_tokens(self):
        """Test consuming tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(5)
        assert bucket.get_tokens() == 5.0

    def test_consume_too_many_tokens(self):
        """Test consuming more tokens than available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert not bucket.consume(15)
        assert bucket.get_tokens() == 10.0

    def test_token_refill(self):
        """Test token refilling over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/second
        bucket.consume(10)
        assert bucket.get_tokens() == 0.0

        time.sleep(0.5)  # Wait 0.5 seconds
        tokens = bucket.get_tokens()
        assert tokens >= 4.0  # Should have refilled ~5 tokens


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_rate_limit_allow(self):
        """Test rate limit allows requests."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.check_rate_limit("client1")

    def test_rate_limit_burst(self):
        """Test burst capacity."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Should allow burst
        for _ in range(5):
            assert limiter.check_rate_limit("client1")

        # Should deny next request
        assert not limiter.check_rate_limit("client1")

    def test_rate_limit_per_client(self):
        """Test rate limiting is per-client."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Exhaust client1's tokens
        for _ in range(5):
            limiter.check_rate_limit("client1")

        # client2 should still have tokens
        assert limiter.check_rate_limit("client2")

    def test_get_remaining(self):
        """Test getting remaining tokens."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        limiter.check_rate_limit("client1")

        remaining = limiter.get_remaining("client1")
        assert remaining == 9.0


class TestInputValidator:
    """Tests for InputValidator."""

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert InputValidator.sanitize_filename("test.md") == "test.md"
        assert InputValidator.sanitize_filename("folder/file.md") == "folder_file.md"
        assert InputValidator.sanitize_filename("test:file?.md") == "test_file_.md"
        assert InputValidator.sanitize_filename("   .test   ") == "test"

    def test_sanitize_filename_empty(self):
        """Test sanitizing empty filename."""
        assert InputValidator.sanitize_filename("") == "untitled"
        assert InputValidator.sanitize_filename("...") == "untitled"

    def test_validate_extension_valid(self):
        """Test valid extension validation."""
        assert InputValidator.validate_extension("test.md", ["md", "txt"])
        assert InputValidator.validate_extension("test.MD", ["md"])

    def test_validate_extension_invalid(self):
        """Test invalid extension rejection."""
        assert not InputValidator.validate_extension("test.exe", ["md", "txt"])
        assert not InputValidator.validate_extension("test", ["md"])

    def test_validate_content_size(self):
        """Test content size validation."""
        small_content = "a" * 100
        assert InputValidator.validate_content_size(small_content, max_size_mb=1)

        large_content = "a" * (11 * 1024 * 1024)  # 11MB
        assert not InputValidator.validate_content_size(large_content, max_size_mb=10)
