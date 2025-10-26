"""
Test script for RAG/semantic search functionality.
"""

import asyncio
from pathlib import Path
from src.mcp_obsidian.server import ObsidianMCPServer


async def test_rag():
    """Test RAG indexing locally."""

    print("🚀 Starting RAG test...\n")

    # Initialize server (make sure you have a config.yaml with RAG enabled)
    print("📋 Loading configuration...")
    try:
        server = ObsidianMCPServer(config_path="config/config.yaml")
    except FileNotFoundError:
        print("❌ Config file not found at config/config.yaml")
        print("💡 Create it from the example: cp config/config.example.yaml config/config.yaml")
        print("   Then edit config.yaml to set your vault path and enable RAG")
        return
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    if not server.rag_engine:
        print("❌ RAG is not enabled in config")
        print("💡 Edit config/config.yaml and set:")
        print("   rag:")
        print("     enabled: true")
        print("     provider: 'ollama'")
        return

    print("✓ RAG engine initialized")
    print(f"  Provider: {server.rag_engine.embedding_provider.get_provider_name()}")
    print(f"  Dimension: {server.rag_engine.embedding_provider.get_dimension()}")

    # Index the vault
    print("\n📚 Indexing vault...")
    try:
        notes_indexed, chunks_added, chunks_updated = await server.rag_engine.index_vault()

        print(f"\n✅ Indexing complete!")
        print(f"  Notes indexed: {notes_indexed}")
        print(f"  Chunks added: {chunks_added}")
        print(f"  Chunks updated: {chunks_updated}")
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        print("\n💡 Common issues:")
        print("  - Is Ollama running? (run: ollama serve)")
        print("  - Is the model pulled? (run: ollama pull nomic-embed-text)")
        print("  - Is your vault path correct in config.yaml?")
        return

    # Get stats
    print(f"\n📊 Index stats:")
    try:
        stats = await server.rag_engine.get_index_stats()
        print(f"  Total documents: {stats.total_documents}")
        print(f"  Total notes: {stats.total_notes}")
        print(f"  Embedding dimension: {stats.embedding_dimension}")
        print(f"  Provider: {stats.provider}")
        print(f"  Index size: {stats.index_size_mb:.2f} MB")
    except Exception as e:
        print(f"  ⚠️ Could not get stats: {e}")

    # Test semantic search
    if notes_indexed > 0:
        print(f"\n🔍 Testing semantic search...")
        try:
            results = await server.rag_engine.semantic_search("test query", k=3)
            print(f"  Found {len(results)} results for 'test query'")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.note.name} (score: {result.score:.3f})")
                print(f"     Preview: {result.chunk_text[:80]}...")
        except Exception as e:
            print(f"  ⚠️ Search error: {e}")
    else:
        print("\n⚠️ No notes indexed, skipping search test")

    print("\n✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_rag())
