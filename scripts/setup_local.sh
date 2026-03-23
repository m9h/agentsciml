#!/bin/bash
# Setup script for local GPU machines (DGX Spark, workstations, etc.)
# Platform-agnostic — works on any Linux box with NVIDIA GPU + uv.
#
# Usage:
#   bash setup_local.sh                  # run directly on the GPU machine
#   ssh user@host 'bash -s' < setup_local.sh  # run remotely via SSH

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/dev}"

echo "============================================"
echo " agentsciml + dmipy Local GPU Setup"
echo "============================================"
echo ""

# ── 1. GPU check ─────────────────────────────────────────────────
echo ">>> Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || {
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
}
echo ""

# ── 2. Install uv ────────────────────────────────────────────────
echo ">>> Checking uv..."
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

# Pin Python 3.12
uv python install 3.12 2>/dev/null || true
echo ""

# ── 3. Clone or update repos ────────────────────────────────────
echo ">>> Setting up repositories in $REPO_DIR..."
mkdir -p "$REPO_DIR"

if [ -d "$REPO_DIR/agentsciml" ]; then
    echo "agentsciml exists, pulling latest..."
    cd "$REPO_DIR/agentsciml" && git pull --ff-only || true
else
    cd "$REPO_DIR"
    git clone https://github.com/m9h/agentsciml.git
fi

if [ -d "$REPO_DIR/dmipy" ]; then
    echo "dmipy exists, pulling latest..."
    cd "$REPO_DIR/dmipy" && git pull --ff-only || true
else
    cd "$REPO_DIR"
    git clone https://github.com/m9h/dmipy.git
fi
echo ""

# ── 4. Install agentsciml ────────────────────────────────────────
echo ">>> Setting up agentsciml..."
cd "$REPO_DIR/agentsciml"
uv sync --dev --python 3.12
echo "agentsciml installed."
echo ""

# ── 5. Install dmipy ─────────────────────────────────────────────
echo ">>> Setting up dmipy..."
cd "$REPO_DIR/dmipy"
uv sync --python 3.12
echo "dmipy installed."
echo ""

# ── 5b. Fix cuDNN version mismatch ───────────────────────────────
# JAX 0.8.1 + jax-cuda13 needs cuDNN >= 9.12.0 but nvidia-cudnn-cu12
# installs 9.10.2 which is too old. Swap to cu13 version.
echo ">>> Fixing cuDNN version..."
cd "$REPO_DIR/dmipy"
CUDNN_CU12_VER=$(uv pip show nvidia-cudnn-cu12 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ -n "$CUDNN_CU12_VER" ]; then
    echo "Removing nvidia-cudnn-cu12==$CUDNN_CU12_VER (incompatible with JAX)"
    uv pip uninstall nvidia-cudnn-cu12 || true
    uv pip install nvidia-cudnn-cu13==9.20.0.48 --reinstall
    echo "cuDNN fixed."
else
    echo "nvidia-cudnn-cu12 not found, checking cu13..."
    uv pip show nvidia-cudnn-cu13 2>/dev/null | grep "^Version:" || echo "WARNING: no cuDNN package found"
fi
echo ""

# ── 6. Verify JAX GPU ────────────────────────────────────────────
echo ">>> Verifying JAX GPU access..."
cd "$REPO_DIR/dmipy"
uv run --no-sync python -c "
import jax
devices = jax.devices()
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU count: {len(gpu_devices)}')
    print('JAX GPU: OK')
else:
    print('WARNING: No GPU devices found by JAX!')
" 2>&1 | grep -v "^ERROR\|^Traceback\|^  File\|ALREADY_EXISTS" || true
echo ""

# ── 7. Check API key ─────────────────────────────────────────────
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY is set."
else
    echo "Set your API key:"
    echo "  echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc"
    echo "  source ~/.bashrc"
fi
echo ""
echo "Run the evolutionary loop:"
echo "  cd $REPO_DIR/agentsciml"
echo "  source ~/.local/bin/env"
echo "  uv run agentsciml run --project $REPO_DIR/dmipy --adapter dmipy --budget 5.0"
echo ""
