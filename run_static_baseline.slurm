#!/bin/bash
#SBATCH --job-name=static-gnn-baseline
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=bader
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/static_baseline_%j.out
#SBATCH --error=logs/static_baseline_%j.err

echo "=============================================="
echo "Static GNN Baseline Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Navigate to project directory
cd ~/temporal-gnn-project

# Activate environment
source set_temporal_gnn

# Create necessary directories
mkdir -p logs results

# Print system info
echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""

echo "GPU info:"
nvidia-smi
echo ""

# Run the static baseline experiment
echo "Starting static GNN baseline training..."
python static_gnn_baseline.py

echo ""
echo "Experiment completed!"
echo "Check results/ directory for outputs"
