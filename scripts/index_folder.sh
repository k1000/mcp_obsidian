#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/index_folder.sh [--config path/to/config.yaml] [--pattern glob]

Indexes only notes matching the provided glob (default: projects/**).
EOF
  exit "${1:-0}"
}

CONFIG_PATH="config/config.yaml"
PATH_PATTERN="projects/**"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --pattern)
      PATH_PATTERN="$2"
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

echo "Indexing pattern '${PATH_PATTERN}' using config: ${CONFIG_PATH}"

uv run python - "$CONFIG_PATH" "$PATH_PATTERN" <<'PY'
import asyncio
import sys

from src.mcp_obsidian.server import ObsidianMCPServer

config_path = sys.argv[1]
path_pattern = sys.argv[2]

server = ObsidianMCPServer(config_path)
asyncio.run(server.rag_engine.index_vault(path_pattern=path_pattern))
PY
