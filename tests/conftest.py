"""
Pytest configuration and fixtures for tests.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest

from mcp_obsidian.vault import VaultManager


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_vault() -> Generator[Path, None, None]:
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)

        # Create some test notes
        (vault_path / "test1.md").write_text(
            """---
title: Test Note 1
tags:
  - test
  - demo
---

# Test Note 1

This is a test note with some content.

#inline-tag

[[test2]]
"""
        )

        (vault_path / "test2.md").write_text(
            """# Test Note 2

Another test note.

Links to [[test1]].
"""
        )

        (vault_path / "folder").mkdir()
        (vault_path / "folder" / "nested.md").write_text(
            """---
title: Nested Note
---

# Nested Note

This is in a subfolder.
"""
        )

        yield vault_path


@pytest.fixture
async def vault_manager(temp_vault: Path) -> AsyncGenerator[VaultManager, None]:
    """Create a VaultManager instance for testing."""
    manager = VaultManager(vault_path=str(temp_vault))
    yield manager
