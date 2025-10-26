#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_server.sh [--config path/to/config.yaml]

Starts the Obsidian MCP server via uv. If no --config is provided, the script
falls back to config/config.yaml when it exists, otherwise the server loads
configuration purely from environment variables.
EOF
  exit "${1:-0}"
}

CONFIG_PATH=""

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

if [[ -z "$CONFIG_PATH" && -f "config/config.yaml" ]]; then
  CONFIG_PATH="config/config.yaml"
fi

if [[ -n "$CONFIG_PATH" ]]; then
  echo "Starting Obsidian MCP server with config: ${CONFIG_PATH}"
  exec uv run python -m mcp_obsidian.server "$CONFIG_PATH"
else
  echo "Starting Obsidian MCP server with environment-only configuration"
  exec uv run python -m mcp_obsidian.server
fi
