.PHONY: test test-unit test-integration test-cov lint fmt clean
.PHONY: gpu-up gpu-down gpu-delete gpu-status gpu-ssh gpu-run gpu-gpus gpu-setup
.PHONY: dgx-setup dgx-run dgx-ssh dgx-build run

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

# ── RunPod GPU cloud ─────────────────────────────────────────────
#
# First-time setup:
#   1. Create account at runpod.io
#   2. Run: runpodctl doctor          (saves API key)
#   3. Run: make gpu-up               (creates A100 pod)
#   4. Run: make gpu-ssh              (SSH into pod)
#   5. Inside pod: make gpu-setup && make gpu-run
#
# Quick reference:
#   make gpu-gpus                     # list available GPUs + prices
#   make gpu-up                       # create A100 pod (default)
#   make gpu-up GPU="NVIDIA RTX A6000"  # use specific GPU
#   make gpu-ssh                      # SSH into pod
#   make gpu-status                   # list pods
#   make gpu-down                     # stop pod (state preserved)
#   make gpu-up                       # restart stopped pod
#   make gpu-delete                   # destroy pod

POD_NAME    ?= agentsciml-dmipy
GPU_ID      ?= NVIDIA A100 80GB PCIe
GPU_COUNT   ?= 1
BUDGET      ?= 5.0
GENERATIONS ?= 20
ADAPTER     ?= dmipy
PROJECT     ?= ~/dmipy
IMAGE       ?= runpod/pytorch:1.0.3-cu1300-torch260-ubuntu2404
SETUP_URL   := https://raw.githubusercontent.com/m9h/agentsciml/main/scripts/setup_runpod.sh

gpu-gpus:
	@runpodctl gpu list 2>/dev/null | python3 -c "\
	import json,sys; \
	gpus = json.load(sys.stdin); \
	print(f'{\"GPU\":<35} {\"VRAM\":>6} {\"Price\":>8} {\"Available\":>10}'); \
	print('-'*65); \
	[print(f'{g[\"id\"]:<35} {g.get(\"memoryInGb\",\"?\"):>4}GB {\"$$\"+str(g.get(\"lowestPrice\",{}).get(\"minimumBidPrice\",\"?\"))[:6]+\"/hr\":>8} {\"yes\" if g.get(\"lowestPrice\",{}).get(\"minimumBidPrice\") else \"no\":>10}') \
	for g in sorted(gpus, key=lambda x: x.get('lowestPrice',{}).get('minimumBidPrice',999))]" \
	2>/dev/null || runpodctl gpu list

gpu-up:
	@# Check if pod already exists and is stopped
	@POD_ID=$$(runpodctl pod list -o json 2>/dev/null | python3 -c "\
	import json,sys; pods=json.load(sys.stdin); \
	matches=[p for p in pods if p.get('name')=='$(POD_NAME)']; \
	print(matches[0]['id'] if matches else '')" 2>/dev/null); \
	if [ -n "$$POD_ID" ]; then \
		echo "Restarting existing pod $$POD_ID..."; \
		runpodctl pod start $$POD_ID; \
	else \
		echo "Creating pod $(POD_NAME) with $(GPU_ID)..."; \
		runpodctl pod create \
			--name "$(POD_NAME)" \
			--gpu-id "$(GPU_ID)" \
			--gpu-count $(GPU_COUNT) \
			--image "$(IMAGE)" \
			--container-disk-in-gb 50 \
			--volume-in-gb 50 \
			--ports "22/tcp,8888/http" \
			--env '{"ANTHROPIC_API_KEY":"$(ANTHROPIC_API_KEY)"}' \
			--ssh; \
	fi
	@echo ""
	@echo "Waiting for pod to be ready..."
	@echo "Run: make gpu-status  (wait for RUNNING)"
	@echo "Then: make gpu-ssh"

gpu-down:
	@POD_ID=$$(runpodctl pod list -o json 2>/dev/null | python3 -c "\
	import json,sys; pods=json.load(sys.stdin); \
	matches=[p for p in pods if p.get('name')=='$(POD_NAME)']; \
	print(matches[0]['id'] if matches else '')" 2>/dev/null); \
	if [ -n "$$POD_ID" ]; then \
		echo "Stopping pod $$POD_ID (state preserved, no GPU charges)..."; \
		runpodctl pod stop $$POD_ID; \
	else \
		echo "No pod named $(POD_NAME) found."; \
	fi

gpu-delete:
	@POD_ID=$$(runpodctl pod list -o json 2>/dev/null | python3 -c "\
	import json,sys; pods=json.load(sys.stdin); \
	matches=[p for p in pods if p.get('name')=='$(POD_NAME)']; \
	print(matches[0]['id'] if matches else '')" 2>/dev/null); \
	if [ -n "$$POD_ID" ]; then \
		echo "Deleting pod $$POD_ID permanently..."; \
		runpodctl pod delete $$POD_ID; \
	else \
		echo "No pod named $(POD_NAME) found."; \
	fi

gpu-status:
	@runpodctl pod list -o json 2>/dev/null | python3 -c "\
	import json,sys; pods=json.load(sys.stdin); \
	print(f'{\"NAME\":<25} {\"STATUS\":<12} {\"GPU\":<30} {\"ID\":<15}'); \
	print('-'*85); \
	[print(f'{p.get(\"name\",\"?\"):<25} {p.get(\"desiredStatus\",\"?\"):<12} {p.get(\"machine\",{}).get(\"gpuDisplayName\",\"?\"):<30} {p[\"id\"]:<15}') \
	for p in pods]" 2>/dev/null || runpodctl pod list

gpu-ssh:
	@POD_ID=$$(runpodctl pod list -o json 2>/dev/null | python3 -c "\
	import json,sys; pods=json.load(sys.stdin); \
	matches=[p for p in pods if p.get('name')=='$(POD_NAME)']; \
	print(matches[0]['id'] if matches else '')" 2>/dev/null); \
	if [ -z "$$POD_ID" ]; then \
		echo "No pod named $(POD_NAME) found. Run: make gpu-up"; \
		exit 1; \
	fi; \
	echo "Getting SSH info for pod $$POD_ID..."; \
	SSH_INFO=$$(runpodctl ssh info $$POD_ID -o json 2>/dev/null); \
	SSH_HOST=$$(echo "$$SSH_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('host',''))" 2>/dev/null); \
	SSH_PORT=$$(echo "$$SSH_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('port',''))" 2>/dev/null); \
	if [ -n "$$SSH_HOST" ] && [ -n "$$SSH_PORT" ]; then \
		echo "Connecting to $$SSH_HOST:$$SSH_PORT..."; \
		ssh -o StrictHostKeyChecking=no -p $$SSH_PORT root@$$SSH_HOST; \
	else \
		echo "SSH info:"; \
		runpodctl ssh info $$POD_ID; \
		echo ""; \
		echo "Connect manually with the info above."; \
	fi

gpu-setup:
	@echo "Running setup (install uv, clone repos, install deps)..."
	curl -fsSL $(SETUP_URL) | bash

gpu-run:
	@echo "Running agentsciml evolutionary loop..."
	source $$HOME/.local/bin/env 2>/dev/null || true; \
	cd $$HOME/agentsciml && git pull --ff-only; \
	uv run agentsciml run \
		--project $(PROJECT) \
		--adapter $(ADAPTER) \
		--budget $(BUDGET) \
		--generations $(GENERATIONS)

# ── DGX Spark / container-based GPU ────────────────────────────
#
# Runs agentsciml in an NGC JAX container on the DGX Spark via SSH.
#
# First-time setup:
#   1. make dgx-build                   (build container on DGX)
#   2. make dgx-run ANTHROPIC_API_KEY=sk-ant-...
#
# Quick reference:
#   make dgx-ssh                        # SSH into DGX
#   make dgx-build                      # build/rebuild container
#   make dgx-run                        # run evolutionary loop in container

DGX_HOST    ?= gx10-dgx-spark.local
DGX_USER    ?= mhough
DGX_KEY     ?= /Users/mhough/Library/Application Support/NVIDIA/Sync/config/nvsync.key
DGX_DIR     ?= ~/dev
DGX_SSH     := ssh -o ConnectTimeout=10 -i "$(DGX_KEY)" $(DGX_USER)@$(DGX_HOST)
JAX_TAG     ?= 26.02-py3
CONTAINER   ?= agentsciml-dmipy

dgx-ssh:
	@$(DGX_SSH)

dgx-build:
	@echo "Syncing Dockerfile to DGX and building container..."
	@scp -i "$(DGX_KEY)" Dockerfile $(DGX_USER)@$(DGX_HOST):$(DGX_DIR)/agentsciml/Dockerfile
	@$(DGX_SSH) '\
		cd $(DGX_DIR)/agentsciml && \
		docker build --build-arg JAX_TAG=$(JAX_TAG) -t $(CONTAINER) .'

dgx-run:
	@echo "Running evolutionary loop on DGX Spark (container)..."
	@$(DGX_SSH) '\
		docker run --rm --gpus all \
			--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
			-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
			$(CONTAINER) \
			--project /workspace/dmipy \
			--adapter $(ADAPTER) \
			--budget $(BUDGET) \
			--generations $(GENERATIONS)'

# ── Local run (on this machine, if it has a GPU) ───────────────

run:
	@echo "Running agentsciml evolutionary loop locally..."
	uv run agentsciml run \
		--project $(PROJECT) \
		--adapter $(ADAPTER) \
		--budget $(BUDGET) \
		--generations $(GENERATIONS)
