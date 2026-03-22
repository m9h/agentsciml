.PHONY: test test-unit test-integration test-cov lint fmt clean
.PHONY: gpu-up gpu-down gpu-status gpu-ssh gpu-setup gpu-run gpu-secret gpu-logs

# ── Local development ─────────────────────────────────────────────

test:
	.venv/bin/python -m pytest tests/ -v

test-unit:
	.venv/bin/python -m pytest tests/ -v -m unit

test-integration:
	.venv/bin/python -m pytest tests/ -v -m integration

test-cov:
	.venv/bin/python -m pytest tests/ -v --cov=agentsciml --cov-report=term-missing

lint:
	.venv/bin/python -m ruff check src/ tests/

fmt:
	.venv/bin/python -m ruff format src/ tests/

clean:
	rm -rf .pytest_cache __pycache__ .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

# ── Brev GPU cloud ────────────────────────────────────────────────
#
# Usage:
#   make gpu-secret KEY=sk-ant-...   # one-time: store Anthropic key
#   make gpu-up                      # create/start A100 instance
#   make gpu-up GPU=l4               # use cheaper L4 instead
#   make gpu-ssh                     # SSH into instance
#   make gpu-run                     # setup + run evolutionary loop
#   make gpu-status                  # check instance status
#   make gpu-down                    # stop (pause) instance — preserves state
#   make gpu-delete                  # fully delete instance

INSTANCE    ?= agentsciml-dmipy
GPU         ?= a100
BUDGET      ?= 5.0
GENERATIONS ?= 20
ADAPTER     ?= dmipy
PROJECT     ?= ~/dmipy
SETUP_URL   := https://raw.githubusercontent.com/m9h/agentsciml/main/scripts/setup_brev.sh

# Map friendly GPU names to brev instance types
GPU_MAP_a100  := a2-highgpu-1g:nvidia-tesla-a100:1
GPU_MAP_l4    := g2-standard-4:nvidia-l4:1
GPU_MAP_t4    := n1-highmem-4:nvidia-tesla-t4:1
GPU_MAP_l40s  := g2-standard-48:nvidia-l40s:1
GPU_MAP_h100  := a3-highgpu-1g:nvidia-h100-80gb:1
GPU_TYPE      = $(GPU_MAP_$(GPU))

gpu-up:
ifndef GPU_TYPE
	$(error Unknown GPU "$(GPU)". Options: a100, l4, t4, l40s, h100)
endif
	@echo "Creating $(INSTANCE) with $(GPU) GPU..."
	@brev ls 2>/dev/null | grep -q "$(INSTANCE).*STOPPED" \
		&& brev start $(INSTANCE) \
		|| brev create $(INSTANCE) -g "$(GPU_TYPE)"
	@echo ""
	@echo "Next: make gpu-ssh   (then: make gpu-run)"

gpu-down:
	@echo "Stopping $(INSTANCE) (state preserved, no charges)..."
	brev stop $(INSTANCE)

gpu-delete:
	@echo "Deleting $(INSTANCE) permanently..."
	brev delete $(INSTANCE)

gpu-status:
	@brev ls 2>&1 | head -10

gpu-ssh:
	brev shell $(INSTANCE)

gpu-secret:
ifndef KEY
	$(error Set your API key: make gpu-secret KEY=sk-ant-...)
endif
	brev secret --name ANTHROPIC_API_KEY --value "$(KEY)" --type variable --scope org
	@echo "API key stored. It will be available as $$ANTHROPIC_API_KEY on all instances."

gpu-setup:
	@echo "Running setup on $(INSTANCE)..."
	@echo "Run this inside the instance (make gpu-ssh first):"
	@echo ""
	@echo "  curl -fsSL $(SETUP_URL) | bash"

gpu-run:
	@echo "Run this inside the instance (make gpu-ssh first):"
	@echo ""
	@echo "  source ~/.local/bin/env"
	@echo "  cd ~/agentsciml && git pull"
	@echo "  uv run agentsciml run --project $(PROJECT) --adapter $(ADAPTER) --budget $(BUDGET) --generations $(GENERATIONS)"

gpu-logs:
	@echo "Run inside the instance to check progress:"
	@echo ""
	@echo "  cd ~/agentsciml && uv run agentsciml status --project $(PROJECT)"
