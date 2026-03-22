#!/bin/bash
# Run this after: brev shell agentsciml-dmipy
# Usage: bash run_on_brev.sh <ANTHROPIC_API_KEY>
#
# Example:
#   bash run_on_brev.sh sk-ant-api03-xxxxx

set -euo pipefail

if [ -z "${1:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Usage: bash run_on_brev.sh <ANTHROPIC_API_KEY>"
    echo "  Or set ANTHROPIC_API_KEY env var first."
    exit 1
fi

export ANTHROPIC_API_KEY="${1:-$ANTHROPIC_API_KEY}"

# 1. Setup (idempotent — safe to re-run)
curl -fsSL https://raw.githubusercontent.com/m9h/agentsciml/main/scripts/setup_brev.sh | bash

# 2. Launch evolutionary loop
cd ~/agentsciml
uv run agentsciml run --project ~/dmipy --adapter dmipy --budget 5.0
