#!/usr/bin/env bash
# run_discovery.sh
# Script to launch the AgenticSciML Stage 1 Discovery Phase for the brain-fwi project.

set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  AgenticSciML: Stage 1 Discovery Phase (brain-fwi surrogate)    ${NC}"
echo -e "${GREEN}=================================================================${NC}"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY is not set in your environment.${NC}"
    echo -e "${YELLOW}Please set it by running: export ANTHROPIC_API_KEY=\"your_key_here\"${NC}"
    echo -e "${YELLOW}Then run this script again.${NC}"
    exit 1
fi

# Ensure we are in the agentsciml directory
cd "$(dirname "$0")"

# Ensure the virtual environment is up to date
echo -e "\n${YELLOW}[1/2] Syncing AgenticSciML dependencies...${NC}"
uv sync --all-extras

# Run the orchestration
echo -e "\n${YELLOW}[2/2] Launching the Swarm Orchestrator...${NC}"
echo -e "This will run 5 generations with a $5.00 API budget."
echo -e "The Agentic Engineer will propose and run experiments in the ../brain-fwi directory."
echo ""

PYTHONPATH=src .venv/bin/python -m agentsciml.cli -v run \
  --project ../brain-fwi \
  --budget 5.0 \
  --generations 5 \
  --knowledge knowledge/brain_fwi_techniques.yaml \
  --debate-rounds 4

echo -e "\n${GREEN}Discovery run complete or stopped. Check agentsciml/autoresearch/tree.json for the solution tree.${NC}"
