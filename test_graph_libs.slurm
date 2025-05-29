#!/bin/bash
#SBATCH --job-name=test-graph-libs
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=bader
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_graph_libs_%j.out
#SBATCH --error=logs/test_graph_libs_%j.err

echo "🚀 Starting graph libraries test on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Load environment
module purge
module load easybuild
module load foss/2022b
module load Python/3.10.8
source ~/temporal-gnn-env/bin/activate

echo "📍 Working directory: $(pwd)"
echo "🐍 Python path: $(which python)"

# Run the test
python test_graph_setup.py

echo "✅ Test completed at $(date)"
