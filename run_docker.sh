#!/usr/bin/env bash
set -euo pipefail

# Build and run Docker container (Linux / macOS)
docker build -t ai-detector:latest .
docker run --rm -p 8000:8000 ai-detector:latest

# After running, open: http://localhost:8000/health