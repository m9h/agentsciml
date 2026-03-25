# Agentic AI: Strategies, Projects, and Paradigms

Research overview of frameworks, minimal implementations, and key thinkers distilling the essence of agentic coding patterns. Compiled 2026-03-23.

---

## Taxonomy of Approaches

| Strategy | Core Idea | Example Projects |
|---|---|---|
| **Memory Architecture** | Give stateless agents persistent identity and memory via filesystem conventions | agent-kernel, Beads |
| **Autonomous Loop** | Single agent iterates through a PRD until all stories complete | Ralph loop variants |
| **Multi-Agent Orchestration** | Specialized agents collaborate via typed protocols | AgenticSciML, Gas Town, CrewAI |
| **Evolutionary Search** | Tree of solutions with mutation, selection, and debate | AgenticSciML |
| **Minimal ReAct** | Reason-Act loop distilled to < 200 lines for teaching | minimal-agent, agents-from-scratch |
| **Personal AI Platform** | Self-hosted assistant with multi-channel gateway + extensible skills | OpenClaw (on Pi Agent runtime) |
| **Graph Orchestration** | Pregel-inspired stateful graphs with cycles, checkpointing, and fine-grained token control | LangGraph |
| **Token-Efficient Swarms** | Neuroscience-inspired hierarchical options + async threading for 50x cost reduction | Lyfe Agents |
| **Autonomous Experiment Loop** | Single agent, one metric, git-backed hypothesis testing, runs overnight | autoresearch (Karpathy Loop) |
| **Model Compression Challenge** | Competitive optimization of L(N) — best model in 16 MB, informing efficient agent backbones | parameter-golf (OpenAI) |

---

## 1. Steve Yegge's Ecosystem (Gas Town + Beads)

**Scale: production multi-agent orchestration (Stage 7-8)**

### Gas Town — Multi-Agent Workspace Manager
- **Repo**: https://github.com/steveyegge/gastown (~12,800 stars)
- **Language**: Go
- **Requires**: Go 1.25+, Git 2.25+, Dolt 1.82.4+, Beads (`bd`) 0.55.4+, tmux 3.0+
- **Concept**: Coordinates 20-30+ parallel AI coding agents on different tasks simultaneously
- **Key abstractions**:
  - **The Mayor**: Your primary Claude Code instance acting as coordinator
  - **Polecats**: Worker agents with persistent identity but ephemeral sessions
  - **Beads/Issues**: Git-backed work items with hash-based IDs (e.g., `gt-abc12`)
  - **Convoys**: Work tracking bundles containing multiple beads assigned to agents
  - **Hooks**: Git worktree-based persistent storage surviving crashes
  - **Refinery**: Per-rig merge queue processor using Bors-style bisecting
  - **Monitoring**: Three-tier watchdog (Witness per-rig, Deacon cross-rig, Dogs maintenance)

### Beads — Agent Memory System
- **Repo**: https://github.com/steveyegge/beads (~19,500 stars)
- Stores agent thoughts, plans, and task dependencies in `.beads/` backed by Dolt (version-controlled SQL)
- Branching code automatically branches agent memory; merging merges memory too
- Blog: ["Introducing Beads"](https://steve-yegge.medium.com/introducing-beads-a-coding-agent-memory-system-637d7d92514a)

### Other Yegge Projects
| Repo | Stars | Description |
|------|-------|-------------|
| [efrit](https://github.com/steveyegge/efrit) | ~413 | Native Elisp coding agent in Emacs |
| [mcp_agent_mail](https://github.com/steveyegge/mcp_agent_mail) | ~39 | "Gmail for coding agents" — inter-agent communication |
| [wasteland](https://github.com/steveyegge/wasteland) | ~11 | Federation protocol for Gas Towns |
| [gastown-otel](https://github.com/steveyegge/gastown-otel) | ~19 | OpenTelemetry observability for Gas Town |

### Yegge's Eight Stages of Developer-Agent Evolution
1. Zero AI — occasional completions/chat
2. IDE agent with permissions on
3. IDE agent, YOLO mode (trust increasing)
4. Wide agent fills IDE; you only review diffs
5. CLI single agent, fully autonomous
6. CLI multi-agent, 3-5 parallel
7. 10+ agents, hand-managed coordination
8. **Build your own orchestrator** (Gas Town lives here)

### Key Writings
- **Book**: *Vibe Coding* (Oct 2025, with Gene Kim) — FAAFO framework
- ["Welcome to Gas Town"](https://steve-yegge.medium.com/welcome-to-gas-town-4f25ee16dd04) (Jan 2026)
- ["The Future of Coding Agents"](https://steve-yegge.medium.com/the-future-of-coding-agents-e9451a84207c) (Jan 2026)
- ["The Anthropic Hive Mind"](https://steve-yegge.medium.com/the-anthropic-hive-mind-d01f768f3d7b) (Feb 2026)
- [Software Engineering Daily interview](https://softwareengineeringdaily.com/2026/02/12/gas-town-beads-and-the-rise-of-agentic-development-with-steve-yegge/)
- [Pragmatic Engineer interview](https://newsletter.pragmaticengineer.com/p/steve-yegge-on-ai-agents-and-the)

---

## 2. Pi Agent + OpenClaw — Personal AI Platform Stack

**Scale: production personal assistant, massive community**

### Pi Agent (Mario Zechner / badlogic)
- **Repo**: https://github.com/badlogic/pi-mono (~27,300 stars)
- **Website**: https://pi.dev
- **Language**: TypeScript (monorepo)
- **Philosophy**: Minimalist terminal-based agent toolkit. Deliberately omits MCP, sub-agents, permission popups, and plan mode — users build these through the extension/skill system.
- **Key packages**:
  - `@mariozechner/pi-ai` — Unified multi-provider LLM API (OpenAI, Anthropic, Google, etc.)
  - `@mariozechner/pi-agent-core` — Agent runtime with tool calling and state management
  - `@mariozechner/pi-coding-agent` — Interactive coding agent CLI
  - `@mariozechner/pi-mom` — Slack bot delegating to pi coding agent
  - `@mariozechner/pi-tui` — Terminal UI with differential rendering
  - `@mariozechner/pi-web-ui` — Web components for AI chat
  - `@mariozechner/pi-pods` — CLI for managing vLLM on GPU pods
- **Features**: 15+ LLM providers, tree-structured conversation history, custom prompt templates/themes
- **Related**: [badlogic/pi-skills](https://github.com/badlogic/pi-skills) (904 stars) — skills compatible with Claude Code and Codex CLI; [Dicklesworthstone/pi_agent_rust](https://github.com/Dicklesworthstone/pi_agent_rust) (591 stars) — Rust reimplementation

### OpenClaw — Self-Hosted Personal AI Assistant
- **Repo**: https://github.com/openclaw/openclaw (~331,800 stars)
- **Website**: https://openclaw.ai
- **Language**: TypeScript
- **Creator**: Peter Steinberger and community (previously "Clawdbot" / "Moltbot")
- **Uses Pi Agent as its underlying agent runtime** (in RPC mode)
- **Key features**:
  - **Multi-channel gateway**: WebSocket control plane routing messages across 20+ platforms (WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Google Chat, IRC, Matrix, LINE, etc.)
  - **Persistent memory**: Learns preferences and context over time
  - **System access**: Browser control, file ops, shell commands, scripts
  - **Extensible skills**: 5,400+ community skills in a registry ([awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills), 41k stars)
  - **Model-agnostic**: Anthropic, OpenAI, Google, or local models
  - **Self-modifying**: Can write and update its own skills
  - **Privacy-first**: Runs locally, data stays on your device

### The OpenClaw Ecosystem
| Project | Stars | Description |
|---------|-------|-------------|
| [zeroclaw-labs/zeroclaw](https://github.com/zeroclaw-labs/zeroclaw) | ~28,500 | Rust-based autonomous AI assistant infrastructure (alternative) |
| [qwibitai/nanoclaw](https://github.com/qwibitai/nanoclaw) | ~25,000 | Lightweight containerized alt, built on Anthropic Agents SDK |
| [HKUDS/nanobot](https://github.com/HKUDS/nanobot) | ~35,700 | "Ultra-Lightweight OpenClaw" |
| [VoltAgent/awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) | ~41,200 | Curated catalog of 5,400+ skills |

### Relationship: Pi Agent vs OpenClaw
- **Pi Agent** = the coding/agent engine (the "brain") — LLM communication, tool calling, state management, code execution
- **OpenClaw** = the personal assistant platform (the "body") — wraps Pi Agent with gateway, messaging integrations, memory, skills system, and multi-platform delivery

---

## 3. LangGraph — Pregel-Inspired Graph Orchestration

**Scale: production stateful agents with fine-grained token control**

- **Repo**: https://github.com/langchain-ai/langgraph (~27,200 stars)
- **Language**: Python (also JS at langchain-ai/langgraphjs)
- **Latest**: v1.1.3 (March 2026)
- **Production users**: Uber, LinkedIn, Klarna, Replit, Elastic
- **Relationship to LangChain**: Separate package (`pip install langgraph`), does NOT require LangChain. But LangChain v1.0 agents run on LangGraph internally.

### Architecture: Graphs with Cycles
Modeled after Google's **Pregel bulk-synchronous parallel** system. Agents are directed graphs with cycles (not DAGs):
1. **Plan** — determine which nodes to execute this super-step
2. **Execute** — run nodes in parallel (same super-step)
3. **Update** — collect outputs, apply to shared state via channel reducers
4. **Repeat** until done

Key abstractions:
- **StateGraph** — typed state schema + nodes + edges
- **Nodes** — Python functions that receive state, return partial updates
- **Edges** — normal, conditional (dynamic routing), or entry/finish
- **Checkpointing** — snapshots after every step (MemorySaver, Sqlite, Postgres) enabling crash recovery, human-in-the-loop, time-travel debugging

### Token Efficiency (Why This Matters)

LangGraph's primary advantage over CrewAI/AutoGen is **zero framework-injected prompt overhead** — you control exactly what goes into the context window.

| Strategy | Mechanism | Savings |
|----------|-----------|---------|
| **Zero system prompt overhead** | No framework-injected tokens (CrewAI adds ~150/agent/request) | ~56% fewer tokens vs CrewAI |
| **Dual-channel ephemeral state** | Separate persistent (conversation) vs ephemeral (reasoning scratchpad) channels | ~60% reduction over 10 steps |
| **Custom message reducers** | `trim_messages`, `RemoveMessage`, milestone-based rolling windows | Prevents O(N^2) growth in loops |
| **LLM summarization** | Periodically compress conversation history | Logarithmic context growth |
| **Node-level caching** (May 2025) | Cache outputs by input hash; skip redundant LLM calls | Eliminates reruns |
| **Checkpoint recovery** | Resume from last success on failure instead of replaying | Avoids re-spending tokens |

**Cost comparison** (3-agent, GPT-4o, 10K requests/day):
- LangGraph: ~$32/day (~800 tokens/request)
- CrewAI: ~$50/day (~1,250 tokens/request) — 56% more
- AutoGen: higher (Group Chat Manager overhead)

### Deep Agents (March 2026)
New harness built on LangGraph: `create_deep_agent(model, tools, system_prompt)` with:
- Built-in planning (todo tool)
- Virtual filesystem (offloads large artifacts out of context)
- Subagent spawning for context isolation
- Cross-thread persistent memory

---

## 4. Lyfe Agents — Token-Efficient Swarms via Neuroscience

**Scale: research prototype, 50x cheaper than Stanford Generative Agents**

- **Repo**: https://github.com/metaconsciousgroup/lyfe-agent-paper (~1 star, research lab)
- **Paper**: [Lyfe Agents: Generative agents for low-cost real-time social interactions](https://arxiv.org/abs/2310.02172), TMLR August 2024
- **Lab**: MetaConscious Group (Guangyu Robert Yang, MIT)
- **Language**: Python (54%), C# (29%), Jupyter (13%)
- **Cost**: ~$0.50/agent/hour vs ~$25/agent/hour for Park et al. (2023) — **50x reduction**

### Core Insight: Resource-Rationality
Mimics how biological brains minimize expensive computation. Three pillars:

#### A. Option-Action Framework (Hierarchical Decision-Making)
- Inspired by hierarchical RL and neuroscience options framework
- `CognitiveController` selects high-level **options** ("talk", "move", "reflect") via LLM
- Once selected, `ActionSelection` executes specific actions **without new LLM calls** until termination (time trigger, repetition detection, or context shift)
- The expensive LLM is only invoked for strategic decisions, not every tick
- **Eliminates ~80-90% of decision-point LLM queries**

#### B. SlowFast Module (Async Threading)
- Core engineering abstraction in `lyfe_agent/slowfast/slowfast.py`
- **Slow function**: LLM call in background thread (expensive, infrequent)
- **Fast function**: Returns cached/default results in main thread (cheap, every tick)
- Decouples agent tick rate from LLM latency — agents stay responsive in real-time

#### C. Summarize-and-Forget Memory
Three-tier memory mirroring cognitive science:
- **Working memory** (capacity 4-5 items): Current context window
- **Recent memory** (embedding + forgetting): Cosine similarity dedup, threshold 0.9
- **Long-term memory** (cluster-then-summarize): Related memories grouped by embedding similarity, each cluster condensed to one summary via single LLM call

### Token Efficiency Summary

| Strategy | Mechanism |
|----------|-----------|
| Infrequent option selection | LLM only called when switching goals |
| Action batching | Multiple actions per option without LLM |
| SlowFast async | Cached fast path; slow LLM on cooldown |
| Memory forgetting | Cosine dedup prevents redundant context |
| Cluster-then-summarize | N memories → ~1 summary (log growth) |
| Proximity filtering | Agents only see nearby events |

### Relevance to Agent-Based Simulation
The SlowFast pattern and Summarize-and-Forget memory are **directly transferable** to computational simulation work. The hierarchical option-action framework maps naturally to agent-based models where agents need to make decisions at multiple timescales.

**Related work**: [Affordable Generative Agents (arXiv:2402.02053)](https://arxiv.org/abs/2402.02053) by Yu et al. (Tencent) — different approach, similar goals, cuts costs to ~31% of baseline.

---

## 5. Agent-Kernel (Oguz Bilgic) — Memory Architecture Pattern

**Scale: zero-code, pure convention**

- **Repo**: https://github.com/oguzbilgic/agent-kernel (~89 stars)
- **What it is**: 4 markdown files. No code. A protocol for giving any AI coding agent persistent state.
- **Key abstractions**:
  - `AGENTS.md` — The "kernel": session lifecycle protocol (read at boot)
  - `IDENTITY.md` — Agent-maintained identity file
  - `KNOWLEDGE.md` — Index of knowledge files
  - `knowledge/` — Mutable semantic memory (current facts)
  - `notes/` — Append-only episodic memory (daily session logs)
- **Persistence**: Git (versioning, sync, conflict resolution for free)
- **Design philosophy**: Radical simplicity. Rejects RAG, vector DBs, embeddings. Flat markdown + git is sufficient for single-agent working memory.
- **Host-agnostic**: Works with Claude Code, Cursor, Codex, etc. — anything that reads project instructions.

### Comparison with Beads
Both solve agent memory across sessions, but at different scales:
- agent-kernel: single agent, markdown files, human-readable
- Beads: multi-agent fleet, Dolt SQL backend, branch-aware graph structure

---

## 6. The Ralph Loop (Geoffrey Huntley) — Autonomous Coding Loop

**Scale: single agent, PRD-driven autonomy**

- **Origin**: Created by Geoffrey Huntley
- **Pattern**: "3 Phases, 2 Prompts, 1 Loop"
  1. **Phase 1 — Requirements**: Define PRD as JSON (stories, gates, status)
  2. **Phase 2 — Planning**: Gap analysis, no implementation
  3. **Phase 3 — Building**: Implement -> test -> commit -> repeat until all stories complete
- **State**: Persists in `.ralph/` files and git. Each iteration starts fresh, reads on-disk state.
- **Curated list**: https://github.com/snwfdhmp/awesome-ralph

### Variants
| Repo | Focus |
|------|-------|
| [iannuttall/ralph](https://github.com/iannuttall/ralph) (839 stars) | Reference implementation; supports codex, claude, droid, opencode |
| [vercel-labs/ralph-loop-agent](https://github.com/vercel-labs/ralph-loop-agent) | Vercel Labs implementation for AI SDK |
| [frankbria/ralph-claude-code](https://github.com/frankbria/ralph-claude-code) | Ralph loop for Claude Code with exit detection |
| [snarktank/ralph](https://github.com/snarktank/ralph) | Runs until all PRD items complete |
| [fstandhartinger/ralph-wiggum](https://github.com/fstandhartinger/ralph-wiggum) | Spec-driven autonomous variant |

---

## 7. AgenticSciML — Evolutionary Multi-Agent Scientific Discovery

**Scale: multi-agent, domain-adapted, evolutionary**

- **Repo**: ~/dev/agentsciml (local)
- **Paper**: Jiang & Karniadakis, CMAME
- **Core pattern**: Evolutionary solution tree with structured multi-agent debate

### Architecture
```
DataAnalyst (Haiku) → Retriever (Haiku) → Debate[Proposer(Sonnet) ↔ Critic(Haiku)]
    → Engineer (Sonnet) → Sandbox execution → Debugger (Haiku, if crash)
    → Score & add to solution tree → Select parents → Repeat
```

### Key differentiators from other frameworks
- **No native tool-calling**: Agents write complete Python experiments; tools are embedded as API surfaces in prompts
- **Typed inter-agent protocols**: Pydantic models (AnalysisReport, MutationProposal, CriticReport, etc.)
- **4-round structured debate**: Rounds 1-2 reasoning-only, round 3 synthesis, round 4 finalization
- **Evolutionary tree**: Exploitation (70%) + exploration (30%) parent selection; MAX_CHILDREN=10 cap
- **Model tier strategy**: Haiku ~80% of calls (analysis), Sonnet ~20% (creative/code gen)
- **Cost-aware**: Budget tracking halts evolution when funds exhausted
- **Domain adapters**: ProjectAdapter interface (~50 lines) to plug in any scientific domain
- **Knowledge base**: YAML technique cards retrieved per mutation (no vector DB needed)
- ~1000 lines core logic total

### Advanced Architectural Patterns (Optimus Integration)

To harden the discovery loop and prevent "shortcut solutions," AgenticSciML incorporates patterns inspired by the **Optimus (2025)** architecture:

1. **Observer-Actor Separation**: Decouples the domain-agnostic Global Planner (agentsciml engine) from the physics-heavy Execution Workers (sandbox projects like qcccm/jaxctrl).
2. **Guard-Rail TDD**: Enforces mandatory `null_test.py` generation for every proposal to mathematically prove physical invariants (e.g., energy bounds, classical limits) *before* expensive simulations.
3. **Adversarial Regularization**: Uses structured N-round debate (configurable up to 6+ rounds) to stress-test hypotheses against known failure modes and "shortcut" hallucinations.
4. **Automated Distillation (Upcoming)**: A `Distiller` agent monitors `results.tsv` for breakthrough scores, automatically extracting the winning logic into new "Technique Cards" in the project's local `knowledge.yaml` to ensure long-term learning.

### Standardized Lab Benchmarks (DynaDojo)

For the **jaxctrl** domain, we adopt the **DynaDojo** benchmarking model to measure scientific progress:
- **Systems**: Standardized dynamical problems (LDS, Pendulum, Lorenz).
- **Controllers**: Differentiable vs. System-ID baselines.
- **Challenges**: Evaluation along axes of **System Complexity** (dimension) and **Sample Efficiency** (data size).
- **Metric**: Normalized control cost across the challenge sweep.

### Comparison with Gas Town
Both are multi-agent, but solve different problems:
- Gas Town: N parallel agents doing independent software engineering tasks, coordinated by a Mayor
- AgenticSciML: N specialized agents in a pipeline collaborating on a single scientific optimization problem

---

## 8. Autoresearch (Andrej Karpathy) — The Karpathy Loop

**Scale: single agent, single GPU, single metric, runs overnight**

- **Repo**: https://github.com/karpathy/autoresearch (~51,700 stars)
- **Created**: March 6, 2026
- **Language**: Python, 630 lines, 3 files
- **Cost**: ~18,000 tokens/cycle, ~$0.10/cycle; overnight 50-round run ~$5

### Architecture: Deliberately Minimal
The entire codebase is 3 files:
- `prepare.py` — Fixed. Data prep, BPE tokenizer, evaluation harness. Never modified.
- `train.py` — The single file the agent edits. Full GPT model + optimizer + training loop.
- `program.md` — The "prompt" steering research direction. Edited by human.

### The Loop
1. Agent reads codebase, forms hypothesis
2. Edits `train.py`
3. Git commit
4. Train for fixed **5-minute wall-clock budget** (`uv run train.py > run.log 2>&1`)
5. Extract `val_bpb` (validation bits per byte) — single metric to minimize
6. If improved: keep commit, advance. If worse/crashed: `git reset`
7. Log to `results.tsv`
8. **NEVER STOP** — loop indefinitely until interrupted

`program.md` instructs: *"The human might be asleep... you are autonomous. If you run out of ideas, think harder."*

### Results
- Karpathy's 48-hour run: 700 experiments, 20 optimizations, 11% speedup
- Shopify CEO Tobi Lutke: 37 experiments overnight, 19% quality improvement
- SkyPilot 16-GPU scaled run: 910 experiments, 2.87% BPB improvement, ~$9 API + ~$300 GPU

### Token Efficiency
- **Minimal codebase**: 3 files fit in one context window — no retrieval overhead
- **Output redirection**: Training logs to file, agent `grep`s only what it needs
- **One-file scope**: Only modifies `train.py` — minimal reasoning per decision
- **Local LLM option**: Community forks run against Qwen 3.5 9B via Ollama for zero API cost

### Relationship to AgenticSciML
Autoresearch is essentially a **single-agent version** of the AgenticSciML pattern:
| | Autoresearch | AgenticSciML |
|---|---|---|
| Agents | 1 (general purpose) | 6 specialized (DataAnalyst, Proposer, Critic, Engineer, Debugger, Retriever) |
| Search strategy | Linear (sequential hypotheses) | Evolutionary tree (exploitation + exploration) |
| Self-critique | None (metric is the judge) | Structured 4-round debate (Proposer ↔ Critic) |
| Knowledge base | None (agent's own knowledge) | YAML technique cards, retrieved per mutation |
| Cost control | Fixed per-experiment time budget | Token budget with cost tracking |

---

## 9. Parameter Golf (OpenAI) — Compression as Efficiency Foundation

**Scale: competitive challenge, informing efficient agent backbones**

- **Repo**: https://github.com/openai/parameter-golf (~3,581 stars)
- **Competition window**: March 18 – April 30, 2026
- **Constraint**: Best language model in **16 MB** (code + compressed weights), trained in **< 10 min on 8xH100**
- **Metric**: Bits per byte (BPB) on FineWeb validation set
- **Prize**: $1M compute credits + recruiting pipeline for OpenAI early-career researchers

### Why It Matters for Agentic AI
Parameter Golf optimizes **L(N)** — the lowest achievable loss for a fixed number of parameters. This is the foundational axis for:
- **Cheaper agent backbones**: Extreme quantization (int5/int6), parameter tying, and novel tokenizers reduce inference cost per token
- **Edge deployment**: 16 MB models can run on mobile/embedded — enabling agents in resource-constrained environments
- **Agent-driven submissions**: Forks like `autoresearch-parameter-golf` use AI agents to automate architecture search for this challenge

### Baseline → SOTA Progress
- **Baseline**: 1.2244 BPB (9-layer transformer, int8 + zlib)
- **Current SOTA**: 1.1233 BPB (8.3% improvement in 5 days) via int6 GPTQ-lite, U-Net skip connections, relu-squared MLP, FlashAttention 3, zstd level 22

### Key Techniques (Relevant to Efficient Agents)
- Aggressive quantization: int5, int6, mixed precision, QAT
- Bigram hash embeddings, novel tokenizers
- Test-time training (LoRA TTT)
- Depth recurrence and parameter tying
- Spectral/orthogonal initialization

### Connection to NanoGPT Ecosystem
Inspired by Keller Jordan's [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt). Together with autoresearch, these three challenges cover the scaling law axes:
- **NanoGPT Speedrun**: L(T) — optimize for time
- **Autoresearch**: L(T) via agent-driven experimentation
- **Parameter Golf**: L(N) — optimize for parameters/compression

---

## 10. Minimal / Educational Agent Implementations

### From-Scratch Tutorials
| Project | Language | Key Feature |
|---------|----------|-------------|
| [langchain-ai/deep-agents-from-scratch](https://github.com/langchain-ai/deep-agents-from-scratch) | Python/Jupyter | 5 progressive notebooks: ReAct → task planning |
| [victordibia/designing-multiagent-systems](https://github.com/victordibia/designing-multiagent-systems) | Python | PicoAgents framework + book; 50+ examples |
| [Antropath/minimal-agent](https://github.com/Antropath/minimal-agent) | Python | Educational ReAct; LiteLLM + DuckDuckGo |
| [pguso/agents-from-scratch](https://github.com/pguso/agents-from-scratch) | Python | Local LLMs only, no frameworks, step-by-step |
| [pguso/ai-agents-from-scratch](https://github.com/pguso/ai-agents-from-scratch) | JS/TS | Same author, Foundation→Composition→Agency→Graphs |
| [Wencho8/ReAct-AI-Agent-from-Scratch-using-DeepSeek](https://github.com/Wencho8/ReAct-AI-Agent-from-Scratch-using-DeepSeek) | Python | DeepSeek-based ReAct with memory |

### Minimal Kernels/Frameworks
| Project | Language | Key Feature |
|---------|----------|-------------|
| [LeonEthan/agentlet](https://github.com/LeonEthan/agentlet) | Python | Minimal loops + explicit context + pluggable capabilities |
| [operand/agency](https://github.com/operand/agency) | Python | Actor model, multiprocessing/multithreading |
| [RajMandaliya/mini-agent](https://github.com/RajMandaliya/mini-agent) | Rust | Async-first, multi-provider, ReAct loop |
| [OasAIStudio/open-agent-sdk](https://github.com/OasAIStudio/open-agent-sdk) | Python | Tools, hooks, skills, subagents — open-source Claude Agent SDK alt |

---

## 11. Key Reference Articles

- **Simon Willison** — ["Designing Agentic Loops"](https://simonwillison.net/2025/Sep/30/designing-agentic-loops/) (Sept 2025): "An LLM agent is something that runs tools in a loop to achieve a goal."
- **Victor Dibia** — ["The Agent Execution Loop"](https://victordibia.com/blog/agent-execution-loop/): Prepare Context → Call Model → Handle Response → Iterate → Return
- **Tweag** — ["Introduction to Agentic Coding"](https://www.tweag.io/blog/2025-10-23-agentic-coding-intro/) (Oct 2025)
- **Temporal** — ["Basic Agentic Loop with Tool Calling"](https://docs.temporal.io/ai-cookbook/agentic-loop-tool-call-openai-python)

---

## Cross-Cutting Themes

1. **Token efficiency is the dominant constraint**: At scale, token cost determines viability. Lyfe Agents achieves 50x reduction via neuroscience-inspired hierarchical options + async threading + forgetting memory. LangGraph achieves ~56% savings over CrewAI via zero framework overhead + ephemeral state channels. AgenticSciML uses model tiering (Haiku 80%, Sonnet 20%) with budget-aware termination. The projects that survive are the ones that treat tokens as a scarce resource.
2. **Git as infrastructure**: Nearly every project uses git for persistence, versioning, and coordination (agent-kernel, Ralph, Beads, Gas Town, AgenticSciML solution tree)
3. **Rejection of heavy frameworks**: The most effective projects avoid CrewAI/AutoGen in favor of plain code + conventions (though LangGraph occupies a middle ground — low-level control without framework opinions)
4. **Memory is the hard problem**: agent-kernel, Beads, Ralph, and Lyfe Agents all center on "how does the agent remember?" — with solutions ranging from flat markdown (agent-kernel) to version-controlled SQL (Beads) to embedding-based forgetting (Lyfe Agents)
5. **Typed protocols beat free-form chat**: AgenticSciML's Pydantic models, Gas Town's structured beads, and LangGraph's typed StateGraph all enforce schema on inter-agent communication
6. **Model tiering**: Use cheap models for routine work, expensive ones for creative/complex work (explicit in AgenticSciML's Haiku/Sonnet split, implicit in Gas Town's Mayor/Polecat hierarchy, and in Lyfe Agents' strategy of reserving LLM calls for high-level option selection only)
7. **Neuroscience as design inspiration**: Lyfe Agents' hierarchical options framework, working/recent/long-term memory tiers, and forgetting/consolidation directly mirror cognitive science models — a pattern increasingly relevant for agent-based simulation research
