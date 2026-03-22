#!/bin/bash
# Create a Brev GPU instance for agentsciml + dmipy
# Run this from your local machine after: brev login
#
# Usage:
#   ./scripts/create_brev_instance.sh              # default: A100 80GB
#   ./scripts/create_brev_instance.sh l4            # cheaper: L4 24GB
#   ./scripts/create_brev_instance.sh a100          # A100 80GB
#   ./scripts/create_brev_instance.sh l40s          # L40S 48GB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

INSTANCE_NAME="agentsciml-dmipy"

# GPU selection with sensible defaults for JAX + diffusion MRI
GPU_CHOICE="${1:-a100}"

case "$GPU_CHOICE" in
    a100)
        GPU_FLAG="nvidia-a100-80gb:1"
        echo "GPU: NVIDIA A100 80GB (recommended for large dmipy models)"
        ;;
    l40s)
        GPU_FLAG="nvidia-l40s:1"
        echo "GPU: NVIDIA L40S 48GB (good balance of cost/performance)"
        ;;
    l4)
        GPU_FLAG="nvidia-l4:1"
        echo "GPU: NVIDIA L4 24GB (budget option, sufficient for small models)"
        ;;
    h100)
        GPU_FLAG="nvidia-h100:1"
        echo "GPU: NVIDIA H100 96GB (maximum performance)"
        ;;
    *)
        echo "Unknown GPU choice: $GPU_CHOICE"
        echo "Options: a100 (default), l40s, l4, h100"
        exit 1
        ;;
esac

echo ""
echo "Creating Brev instance: $INSTANCE_NAME"
echo "Setup script: $REPO_ROOT/scripts/setup_brev.sh"
echo ""

# Check auth
if ! brev ls &> /dev/null; then
    echo "ERROR: Not authenticated. Run: brev login"
    exit 1
fi

# Create instance
brev create "$INSTANCE_NAME" -g "$GPU_FLAG"

echo ""
echo "Instance created. Next steps:"
echo "  1. Wait for it to be ready:  brev ls"
echo "  2. SSH in:                   brev shell $INSTANCE_NAME"
echo "  3. Run setup:                curl -fsSL https://raw.githubusercontent.com/m9h/agentsciml/main/scripts/setup_brev.sh | bash"
echo "     Or copy and run locally:  scp scripts/setup_brev.sh $INSTANCE_NAME:~/setup_brev.sh"
echo "                               brev shell $INSTANCE_NAME -- bash ~/setup_brev.sh"
echo ""
