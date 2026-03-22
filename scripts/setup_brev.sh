#!/bin/bash
# Setup script for NVIDIA Brev GPU instance
# Run this after SSH'ing into the Brev instance:
#   brev shell agentsciml-dmipy
#   bash ~/setup_brev.sh
#
# Or pass to brev create:
#   brev create agentsciml-dmipy -g "nvidia-a100-80gb:1" && brev shell agentsciml-dmipy

set -euo pipefail

echo "============================================"
echo " agentsciml + dmipy Brev GPU Setup"
echo "============================================"
echo ""

# ── 1. System checks ───────────────────────────────────────────────
echo ">>> Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi
echo ""

echo ">>> Checking CUDA..."
if [ -d /usr/local/cuda ]; then
    ls -d /usr/local/cuda* 2>/dev/null | head -5
    echo "CUDA_HOME: ${CUDA_HOME:-/usr/local/cuda}"
else
    echo "WARNING: /usr/local/cuda not found."
fi
echo ""

# ── 2. Install system deps ────────────────────────────────────────
echo ">>> Installing system dependencies (cmake, build-essential)..."
sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential > /dev/null 2>&1 || \
    echo "NOTE: apt-get failed — cmake may need manual install."
echo ""

# ── 3. Install uv + pin Python 3.12 ──────────────────────────────
echo ">>> Installing uv..."
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

echo ">>> Pinning Python 3.12 (3.14 too new for some deps)..."
uv python install 3.12
echo ""

# ── 4. Clone repos ─────────────────────────────────────────────────
echo ">>> Cloning repositories..."
cd "$HOME"

if [ -d "$HOME/agentsciml" ]; then
    echo "agentsciml already exists, pulling latest..."
    cd "$HOME/agentsciml" && git pull --ff-only || true
else
    git clone https://github.com/m9h/agentsciml.git
fi

if [ -d "$HOME/dmipy" ]; then
    echo "dmipy already exists, pulling latest..."
    cd "$HOME/dmipy" && git pull --ff-only || true
else
    git clone https://github.com/m9h/dmipy.git
fi
echo ""

# ── 5. Install agentsciml ──────────────────────────────────────────
echo ">>> Setting up agentsciml..."
cd "$HOME/agentsciml"
uv sync --dev --python 3.12
echo "agentsciml installed."
echo ""

# ── 6. Install dmipy + JAX GPU ─────────────────────────────────────
echo ">>> Setting up dmipy..."
cd "$HOME/dmipy"
uv sync --python 3.12
echo "dmipy installed."
echo ""

# Install JAX with CUDA support (if not already in dmipy deps)
echo ">>> Ensuring JAX GPU support..."
cd "$HOME/dmipy"
uv pip install --upgrade "jax[cuda12]" 2>/dev/null || \
    uv pip install --upgrade "jax[cuda12_pip]" 2>/dev/null || \
    echo "NOTE: JAX CUDA install requires manual config. Check dmipy pyproject.toml."
echo ""

# ── 7. Verify GPU access from Python ───────────────────────────────
echo ">>> Verifying JAX GPU access..."
cd "$HOME/dmipy"
uv run python -c "
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
    print('You may need to install jax[cuda12] manually.')
" || echo "WARNING: JAX GPU verification failed."
echo ""

# ── 8. Set up secrets ──────────────────────────────────────────────
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Set your Anthropic API key:"
echo "     export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "     Or use brev secrets (persists across restarts):"
echo "     brev secret set ANTHROPIC_API_KEY"
echo ""
echo "  2. Run the evolutionary loop:"
echo "     cd ~/agentsciml"
echo "     uv run agentsciml run --project ~/dmipy --adapter dmipy --budget 5.0"
echo ""
echo "  3. Check status:"
echo "     cd ~/agentsciml"
echo "     uv run agentsciml status --project ~/dmipy"
echo ""
echo "  4. Monitor GPU usage:"
echo "     watch -n 1 nvidia-smi"
echo ""
