#!/bin/bash
# .brev/setup.sh — Auto-runs when a Brev instance starts
# This is Brev's convention for instance initialization.
#
# To use: push this repo to GitHub, then create an instance from the repo URL:
#   brev create agentsciml-dmipy -g "nvidia-a100-80gb:1"
#   (Brev auto-detects .brev/setup.sh and runs it)

set -euo pipefail

echo "============================================"
echo " Brev auto-setup: agentsciml + dmipy"
echo "============================================"

# ── 1. Install uv ──────────────────────────────────────────────────
echo ">>> Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 2. Clone dmipy (agentsciml is already present as the repo) ────
echo ">>> Cloning dmipy..."
cd "$HOME"
if [ ! -d "$HOME/dmipy" ]; then
    git clone https://github.com/m9h/dmipy.git
else
    cd "$HOME/dmipy" && git pull --ff-only || true
fi

# ── 3. Install agentsciml ──────────────────────────────────────────
echo ">>> Installing agentsciml..."
# When Brev clones from a repo URL, the repo is at the workspace root
AGENTSCIML_DIR="${BREV_WORKSPACE:-$HOME/agentsciml}"
if [ -d "$AGENTSCIML_DIR" ]; then
    cd "$AGENTSCIML_DIR"
    uv sync --dev
    echo "agentsciml installed from $AGENTSCIML_DIR"
else
    echo "WARNING: agentsciml not found at $AGENTSCIML_DIR"
    cd "$HOME"
    git clone https://github.com/m9h/agentsciml.git
    cd "$HOME/agentsciml"
    uv sync --dev
fi

# ── 4. Install dmipy + JAX GPU ─────────────────────────────────────
echo ">>> Installing dmipy..."
cd "$HOME/dmipy"
uv sync

echo ">>> Installing JAX with CUDA support..."
uv pip install --upgrade "jax[cuda12]" 2>/dev/null || \
    uv pip install --upgrade "jax[cuda12_pip]" 2>/dev/null || \
    echo "NOTE: JAX CUDA auto-install failed. Will need manual setup."

# ── 5. Verify ──────────────────────────────────────────────────────
echo ">>> Verifying setup..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: not detected yet"

cd "$HOME/dmipy"
uv run python -c "
import jax
print(f'JAX {jax.__version__} — devices: {jax.devices()}')
" 2>/dev/null || echo "JAX verification deferred (may need instance restart)."

echo ""
echo "============================================"
echo " Auto-setup complete!"
echo " Set ANTHROPIC_API_KEY, then run:"
echo "   cd ~/agentsciml && uv run agentsciml run --project ~/dmipy --adapter dmipy --budget 5.0"
echo "============================================"
