import asyncio
from src.mcp_obsidian.server import ObsidianMCPServer

async def search():
    server = ObsidianMCPServer(config_path="config/config.yaml")
    
    # Try different queries
    queries = [
        "machine learning and artificial intelligence",
        "trading strategies",
        "music production",
        "cryptocurrency blockchain"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = await server.rag_engine.semantic_search(query, k=5)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.note.name} (score: {r.score:.3f})")

asyncio.run(search())
