#!/usr/bin/env bash
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <MODEL_URL> [sha256]"
  exit 1
fi

URL="$1"
SHA="${2:-}"

if [ -n "$SHA" ]; then
  python download_model.py "$URL" --sha256 "$SHA"
else
  python download_model.py "$URL"
fi
