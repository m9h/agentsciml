# AgenticSciML Conventions

## Overview
This repository implements a multi-agent evolutionary framework for SciML. 

## Key Conventions
- **Adapters**: All project-specific logic must be encapsulated in a `ProjectAdapter` subclass in `src/agentsciml/adapters/`.
- **Knowledge Base**: Curated techniques should be stored as YAML in `knowledge/`.
- **Sandbox**: Experiments are executed via the `sandbox.py` module, supporting local, Slurm, and Modal backends.
- **Debate Protocol**: Uses a multi-round debate between Proposer and Critic agents before code implementation.

## Active Project: Brain-FWI
- **Adapter**: `BrainFWIAdapter`
- **Goal**: Minimize `brain_rmse` for transcranial ultrasound FWI.
- **Hardware**: Dispatches to Modal H100 for 3D JAX simulations.
- **Agent Config**: Currently running All-Opus swarm for deep discovery.
