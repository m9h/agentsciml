#!/bin/bash
#SBATCH --job-name=agentsciml
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.log
#
# Submit from the DGX:
#   sbatch scripts/slurm_run.sh
#
# Or with overrides:
#   BUDGET=10.0 GENERATIONS=30 sbatch scripts/slurm_run.sh

CONTAINER="${CONTAINER:-agentsciml-dmipy}"
BUDGET="${BUDGET:-5.0}"
GENERATIONS="${GENERATIONS:-20}"
ADAPTER="${ADAPTER:-dmipy}"

echo "============================================"
echo " AgenticSciML Slurm Job"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " GPU:         $CUDA_VISIBLE_DEVICES"
echo " Container:   $CONTAINER"
echo " Budget:      \$$BUDGET"
echo " Generations: $GENERATIONS"
echo "============================================"
echo ""

docker run --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    "${CONTAINER}" \
    --project /workspace/dmipy \
    --adapter "${ADAPTER}" \
    --budget "${BUDGET}" \
    --generations "${GENERATIONS}"
