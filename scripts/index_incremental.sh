#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/index_incremental.sh [--config path/to/config.yaml]

Runs the incremental RAG index build (no force re-index).
EOF
  exit "${1:-0}"
}

CONFIG_PATH="config/config.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Incremental indexing using config: ${CONFIG_PATH}"

uv run python - "$CONFIG_PATH" <<'PY'
import asyncio
import sys

from src.mcp_obsidian.server import ObsidianMCPServer

config_path = sys.argv[1]

server = ObsidianMCPServer(config_path)
asyncio.run(server.rag_engine.index_vault(force_reindex=False))
PY
