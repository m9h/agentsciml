# AgenticSciML

Multi-agent evolutionary framework for automated scientific machine learning discovery.

Based on [AgenticSciML](https://arxiv.org/abs/2511.07262) (Jiang & Karniadakis, 2025) with swarm cost optimization from [Flexible Swarm Learning](https://arxiv.org/abs/2510.06349) (Samadi & Schuppert, 2025).

## What it does

AgenticSciML coordinates specialized LLM agents that collaboratively propose, critique, and refine scientific computing experiments through structured debate and evolutionary search. Instead of a human manually tuning parameters and architectures, the system:

1. **Analyzes** experimental history to identify patterns and unexplored regions
2. **Retrieves** relevant techniques from a curated knowledge base
3. **Debates** — a Proposer reasons about what to try; a Critic challenges the reasoning
4. **Implements** the proposed mutation as executable experiment code
5. **Evaluates** results in a sandboxed subprocess
6. **Evolves** — maintains a branching solution tree, balancing exploitation of the best solutions with exploration of new directions

The evolutionary tree typically discovers strategies that outperform single-shot approaches by orders of magnitude, including novel combinations not present in the knowledge base.

### Swarm cost optimization

Following the swarm learning insight that ensembles of smaller specialized agents can outperform monolithic large models, AgenticSciML routes ~80% of calls to fast/cheap models (Haiku) and reserves expensive reasoning models (Sonnet/Opus) for creative proposal generation and code writing. Typical cost per evolutionary generation: **$0.05–0.50**.

## Installation

```bash
git clone https://github.com/m9h/agentsciml.git
cd agentsciml
uv sync --all-extras
```

Requires an `ANTHROPIC_API_KEY` environment variable.

## Usage

```bash
# Run evolutionary search on a project
agentsciml run --project ~/dev/quantum-cognition --budget 5.0 --generations 20

# Check solution tree status
agentsciml status --project ~/dev/quantum-cognition
```

## Architecture

```
orchestrator.py     Evolutionary loop (init → root → tree expansion)
agents.py           8 agent roles with model-tier routing
tree.py             Solution tree with exploitation/exploration parent selection
knowledge.py        YAML-based technique knowledge base
sandbox.py          Subprocess execution with timeout and RESULT| parsing
cost.py             Token tracking, budget caps, escalation rules
protocols.py        Pydantic models for typed inter-agent document passing
adapters/           Project-specific bridges (one per target project)
```

### Agent roles

| Agent | Model | Purpose |
|-------|-------|---------|
| DataAnalyst | Haiku | Summarize results history, identify patterns |
| Retriever | Haiku | Select 0–1 techniques from knowledge base |
| Proposer | Sonnet | Creative reasoning via structured debate |
| Critic | Haiku | Challenge proposals, find flaws |
| Engineer | Sonnet | Write valid experiment code |
| Debugger | Haiku | Fix crashes from stderr |
| ResultAnalyst | Haiku | Evaluate and compare results |
| SelectorEnsemble | 3× Haiku | Diverse voting for parent selection |

## Application areas

AgenticSciML is designed for JAX-based scientific computing projects with a `loss → gradient → optimize` loop. Each project needs a thin adapter (~50 lines) mapping its experiment interface to the framework.

### Quantum cognition — [qcccm](https://github.com/m9h/quantum-cognition)

Quantum cognition library exploiting the Hamiltonian isomorphism between disordered magnets and multi-agent social systems. JAX + PennyLane.

**AgenticSciML targets:** VQE ansatz discovery, QAOA depth-vs-performance tradeoffs, solver meta-selection (when to switch from PIMC to VQE to QAOA), Trotter number optimization, transverse field annealing schedules. The primary metric — `quantum_advantage = (E_classical - E_quantum) / |E_exact|` — measures whether quantum methods find lower-energy social equilibria than classical Monte Carlo.

### Differentiable control — [jaxctrl](https://github.com/m9h/jaxctrl)

Differentiable control theory in JAX: Lyapunov/Riccati solvers, LQR, system identification (SINDy/DMD/Koopman), tensor-based control, hypergraph controllability.

**AgenticSciML targets:** SINDy hyperparameter tuning (sparsity threshold, polynomial degree, Fourier harmonics), operator basis discovery for Koopman learning, multi-system joint identification. The framework can evolve governing equation ansatze — which combination of polynomial and Fourier features best recovers unknown dynamics from time-series data.

### Active inference — [alf](https://github.com/m9h/alf)

Standalone JAX-native active inference library with differentiable HMM learning, expected free energy computation, hierarchical inference, and deep generative models.

**AgenticSciML targets:** Generative model structure search (A/B/C/D matrix sparsity and rank), EFE horizon optimization, precision scheduling, learning rate meta-optimization. The key question: in which environments does active inference exhibit qualitatively different behavior from RL baselines (information-seeking, risk-sensitivity, habit formation)?

### Cortical culture simulation — [bl1](https://github.com/m9h/bl1)

In-silico cortical culture simulator (DishBrain-inspired) — JAX-based spiking neural network with STDP, virtual MEA, and closed-loop game experiments. Simulates dissociated cortical cultures on multi-electrode arrays at 5.3× realtime for 10K neurons on A100.

**AgenticSciML targets:** STDP rule parameter sweeps (timing windows, learning rates), network topology evolution (connectivity patterns that produce target firing statistics), pharmacological intervention optimization (which virtual drug combinations maximize information transfer), and closed-loop stimulation protocol discovery. The evolutionary search can explore the space of plasticity rules and connectivity motifs to find cultures that learn game tasks fastest.

### Evolutionary robotics — evo-embodied

GPU-accelerated evolutionary robotics environment using MuJoCo-MJX and JAX. Replaces sequential PyBullet simulations with vectorized GPU evaluation, achieving 100–1000× speedup for neuroevolution.

**AgenticSciML targets:** Fitness function design, morphology parameterization, neural controller architecture search. The framework can evolve experiment configurations — which combinations of body plan parameters, mutation operators, and selection pressures produce the most diverse and capable locomotion strategies. Particularly suited because the fast MJX evaluation loop means the GPU compute bottleneck is small relative to LLM reasoning time.

### Diffusion MRI microstructure — [dmipy](https://github.com/AthenaEPI/dmipy)

Open-source toolbox for reproducible estimation of brain tissue microstructure from diffusion MRI. Multi-compartment modeling with modular architecture for custom tissue models.

**AgenticSciML targets:** Compartment model selection (which combination of intra-axonal, extra-axonal, and CSF compartments best fits a given acquisition scheme), orientation distribution optimization, and acquisition protocol design. The evolutionary search can explore the combinatorial space of multi-compartment models — testing whether novel compartment combinations improve parameter recovery on specific tissue types or pathologies.

## Writing a project adapter

```python
from agentsciml.adapters.base import ProjectAdapter

class MyProjectAdapter(ProjectAdapter):
    def get_context(self) -> str:
        """Return project description and research goals."""

    def get_results_history(self) -> str:
        """Return accumulated experimental results (TSV, CSV, etc.)."""

    def get_current_experiment(self) -> str:
        """Return current experiment code."""

    def get_available_api(self) -> str:
        """Return the API surface the Engineer agent must use."""

    def get_metric_name(self) -> str:
        """Primary metric name (e.g. 'quantum_advantage')."""

    def get_result_metric_key(self) -> str:
        """Key in RESULT| lines for the primary metric."""

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract primary metric from experiment output."""
```

## References

- Jiang, Q. & Karniadakis, G. E. (2025). AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning. [arXiv:2511.07262](https://arxiv.org/abs/2511.07262)
- Samadi, M. E. & Schuppert, A. (2025). Flexible Swarm Learning May Outpace Foundation Models in Essential Tasks. [arXiv:2510.06349](https://arxiv.org/abs/2510.06349)

## License

MIT
