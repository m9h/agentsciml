#!/usr/bin/env bash
# monitor_discovery.sh
# Script to monitor the live progress of the AgenticSciML evolutionary tree.

# Ensure we are in the agentsciml directory
cd "$(dirname "$0")"

echo "Monitoring AgenticSciML status for brain-fwi..."
echo "Press Ctrl+C to stop."
echo ""

while true; do
  clear
  echo "Last update: $(date)"
  echo "-----------------------------------------------------------------"
  # Run the status CLI command
  PYTHONPATH=src .venv/bin/python -m agentsciml.cli status --project ../brain-fwi
  echo "-----------------------------------------------------------------"
  sleep 10
done
