#!/bin/bash
#SBATCH --job-name=temporal-gnn-anomaly
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=bader
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/temporal_anomaly_%j.out
#SBATCH --error=logs/temporal_anomaly_%j.err

echo "=============================================="
echo "Temporal GNN Anomaly Detection Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Navigate to project directory
cd ~/temporal-gnn-project

# Activate environment
source set_temporal_gnn

# Create necessary directories
mkdir -p logs results src/models/temporal

# Copy temporal modules to appropriate locations
echo "Setting up temporal modules..."

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

# Verify data availability
echo "Checking data availability..."
ls -la data/synthetic/
echo ""

# Run the temporal anomaly detection experiment
echo "Starting temporal anomaly detection experiment..."
echo "This will:"
echo "1. Load synthetic temporal graph data"
echo "2. Train temporal memory modules (TGN/DyRep/JODIE inspired)"
echo "3. Compare with static DOMINANT baseline"
echo "4. Generate comprehensive results and visualizations"
echo ""

python temporal_anomaly_detector.py

# Check results
echo ""
echo "Experiment completed! Checking results..."
echo ""

if [ -d "results" ]; then
    echo "Results directory contents:"
    ls -la results/
    echo ""
    
    if [ -f "results/temporal_comparison_results.txt" ]; then
        echo "=========================================="
        echo "TEMPORAL ANOMALY DETECTION RESULTS"
        echo "=========================================="
        cat results/temporal_comparison_results.txt
        echo ""
    fi
    
    if [ -f "results/temporal_training_curves.png" ]; then
        echo "✅ Training curves generated: temporal_training_curves.png"
    fi
    
    if [ -f "results/temporal_vs_static_comparison.png" ]; then
        echo "✅ Performance comparison generated: temporal_vs_static_comparison.png"
    fi
    
    if [ -f "results/temporal_anomaly_timeline.png" ]; then
        echo "✅ Temporal timeline generated: temporal_anomaly_timeline.png"
    fi
else
    echo "❌ Results directory not found!"
fi

echo ""
echo "=============================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo ""

# Check if experiment was successful
if [ -f "results/temporal_comparison_results.txt" ]; then
    echo "✅ Experiment completed successfully!"
    echo "📊 Check results/ directory for detailed analysis"
    echo "📈 Visualizations available for review"
    echo ""
    echo "🎯 Key Files Generated:"
    echo "   • temporal_comparison_results.txt - Numerical results"
    echo "   • temporal_training_curves.png - Training progress"
    echo "   • temporal_vs_static_comparison.png - Performance comparison"
    echo "   • temporal_anomaly_timeline.png - Temporal anomaly patterns"
else
    echo "❌ Experiment may have failed"
    echo "📋 Check error logs for debugging information"
fi

echo ""
echo "=============================================="
