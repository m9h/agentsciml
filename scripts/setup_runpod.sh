#!/bin/bash
# Setup script for RunPod GPU pod
# Run inside the pod after: make gpu-ssh
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/m9h/agentsciml/main/scripts/setup_runpod.sh | bash
#   # or: make gpu-setup

set -euo pipefail

echo "============================================"
echo " agentsciml + dmipy RunPod GPU Setup"
echo "============================================"
echo ""

# ── 0. Extract API key from container env (RunPod SSH fix) ───────
# RunPod container env vars are NOT available in SSH sessions.
# Extract from the init process environment and persist to ~/.bashrc.
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f /proc/1/environ ]; then
    export ANTHROPIC_API_KEY=$(cat /proc/1/environ | tr '\0' '\n' | grep '^ANTHROPIC_API_KEY=' | cut -d= -f2-)
fi

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    # Persist to ~/.bashrc so future SSH sessions have it
    if ! grep -q 'ANTHROPIC_API_KEY' ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc <<'BASHRC_EOF'

# ── AgenticSciML: extract RunPod container env vars for SSH sessions ──
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f /proc/1/environ ]; then
    export ANTHROPIC_API_KEY=$(cat /proc/1/environ | tr '\0' '\n' | grep '^ANTHROPIC_API_KEY=' | cut -d= -f2-)
fi
BASHRC_EOF
        echo ">>> ANTHROPIC_API_KEY extracted and persisted to ~/.bashrc"
    else
        echo ">>> ANTHROPIC_API_KEY already in ~/.bashrc"
    fi
else
    echo ">>> WARNING: ANTHROPIC_API_KEY not found in environment or /proc/1/environ"
fi
echo ""

# ── 1. GPU check ─────────────────────────────────────────────────
echo ">>> Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not found"
echo ""

# ── 2. Install uv ────────────────────────────────────────────────
echo ">>> Installing uv..."
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

# Pin Python 3.12
uv python install 3.12 2>/dev/null || true
echo ""

# ── 3. Clone repos into /workspace (persistent volume) ──────────
echo ">>> Cloning repositories..."
cd /workspace

if [ -d "/workspace/agentsciml" ]; then
    echo "agentsciml exists, pulling latest..."
    cd /workspace/agentsciml && git pull --ff-only || true
else
    git clone https://github.com/m9h/agentsciml.git
fi

if [ -d "/workspace/dmipy" ]; then
    echo "dmipy exists, pulling latest..."
    cd /workspace/dmipy && git pull --ff-only || true
else
    git clone https://github.com/m9h/dmipy.git
fi
echo ""

# ── 4. Install agentsciml ────────────────────────────────────────
echo ">>> Setting up agentsciml..."
cd /workspace/agentsciml
uv sync --dev --python 3.12
echo "agentsciml installed."
echo ""

# ── 5. Install dmipy ─────────────────────────────────────────────
echo ">>> Setting up dmipy..."
cd /workspace/dmipy
uv sync --python 3.12
echo "dmipy installed."
echo ""

# ── 5b. Fix cuDNN version for JAX 0.8.1 + jax-cuda13 ────────────
# dmipy pulls in nvidia-cudnn-cu12 (9.10.2) which is too old.
# JAX 0.8.1 with jax-cuda13 needs cuDNN >= 9.12.0.
echo ">>> Fixing cuDNN version (cu12 -> cu13)..."
cd /workspace/dmipy
uv pip uninstall nvidia-cudnn-cu12 2>/dev/null || true
uv pip install nvidia-cudnn-cu13==9.20.0.48 --reinstall
echo "cuDNN fixed: nvidia-cudnn-cu13==9.20.0.48"
echo ""

# ── 6. Verify JAX GPU ────────────────────────────────────────────
echo ">>> Verifying JAX GPU access..."
cd /workspace/dmipy
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
" 2>&1 | grep -v "^ERROR\|^Traceback\|^  File\|ALREADY_EXISTS" || true
echo ""

# ── 7. Done ──────────────────────────────────────────────────────
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY is set."
else
    echo "Set your API key:  export ANTHROPIC_API_KEY='sk-ant-...'"
fi
echo ""
echo "Run the evolutionary loop:"
echo "  cd /workspace/agentsciml"
echo "  source ~/.local/bin/env"
echo "  uv run agentsciml run --project /workspace/dmipy --adapter dmipy --budget 5.0"
echo ""
