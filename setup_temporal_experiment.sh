#!/bin/bash
# Setup script for Temporal GNN Anomaly Detection Experiment
# Run this from ~/temporal-gnn-project/

echo "=============================================="
echo "Setting up Temporal GNN Anomaly Detection"
echo "=============================================="

# Create enhanced directory structure
echo "Creating directory structure..."
mkdir -p src/models/temporal
mkdir -p src/evaluation/temporal
mkdir -p configs/temporal
mkdir -p experiments/temporal
mkdir -p results/temporal
mkdir -p logs/temporal

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/models/temporal/__init__.py
touch src/evaluation/__init__.py
touch src/evaluation/temporal/__init__.py

echo "Directory structure created:"
tree src/ configs/ experiments/ results/ 2>/dev/null || ls -la src/ configs/ experiments/ results/

echo ""
echo "=============================================="
echo "TEMPORAL EXPERIMENT FILES TO CREATE"
echo "=============================================="

echo ""
echo "📁 File Organization:"
echo "~/temporal-gnn-project/"
echo "├── temporal_memory_module.py          # Core temporal components"
echo "├── temporal_anomaly_detector.py       # Main integration script"
echo "├── run_temporal_experiment.slurm      # SLURM job script"
echo "├── static_gnn_baseline.py             # Existing static baseline"
echo "├── src/"
echo "│   ├── models/"
echo "│   │   └── temporal/"
echo "│   │       ├── __init__.py"
echo "│   │       ├── memory_modules.py      # Memory components"
echo "│   │       ├── temporal_encoder.py    # Temporal GCN encoder"
echo "│   │       └── trajectory_predictor.py # Prediction components"
echo "│   └── evaluation/"
echo "│       └── temporal/"
echo "│           ├── __init__.py"
echo "│           ├── metrics.py             # Temporal-specific metrics"
echo "│           └── visualization.py       # Advanced visualizations"
echo "├── configs/"
echo "│   └── temporal/"
echo "│       ├── temporal_config.yaml       # Configuration file"
echo "│       └── ablation_configs/          # For ablation studies"
echo "├── results/"
echo "│   └── temporal/                      # Temporal experiment results"
echo "└── logs/"
echo "    └── temporal/                      # Temporal experiment logs"

echo ""
echo "=============================================="
echo "NEXT STEPS"
echo "=============================================="

echo ""
echo "1. 📝 Save the temporal modules:"
echo "   • Copy temporal_memory_module.py to project root"
echo "   • Copy temporal_anomaly_detector.py to project root"
echo "   • Copy run_temporal_experiment.slurm to project root"

echo ""
echo "2. 🔧 Verify dependencies:"
echo "   • Ensure static_gnn_baseline.py is in project root"
echo "   • Verify synthetic data is in data/synthetic/"
echo "   • Check that conda environment is activated"

echo ""
echo "3. 🚀 Run the experiment:"
echo "   sbatch run_temporal_experiment.slurm"

echo ""
echo "4. 📊 Monitor progress:"
echo "   • squeue -u \$USER"
echo "   • tail -f logs/temporal_anomaly_*.out"
echo "   • Check results/temporal/ for outputs"

echo ""
echo "=============================================="
echo "EXPECTED RESULTS"
echo "=============================================="

echo ""
echo "🎯 Performance Targets:"
echo "   • Temporal AUC: > 0.80 (vs Static 0.33)"
echo "   • Improvement: > 2.4x AUC increase"
echo "   • All anomaly types detected (T=15, 30, 45)"
echo "   • Training convergence < 50 epochs"

echo ""
echo "📈 Generated Outputs:"
echo "   • Temporal vs Static comparison metrics"
echo "   • Training curves and validation plots"
echo "   • Temporal anomaly timeline visualization"
echo "   • Comprehensive performance analysis"

echo ""
echo "🔍 Key Insights Expected:"
echo "   • Memory modules capture normal patterns"
echo "   • Temporal attention improves detection"
echo "   • Prediction errors identify anomalies early"
echo "   • Multi-scale modeling handles different anomaly types"

echo ""
echo "=============================================="
echo "TROUBLESHOOTING"
echo "=============================================="

echo ""
echo "💡 Common Issues & Solutions:"
echo ""
echo "1. Import Errors:"
echo "   • Ensure all .py files are in project root"
echo "   • Check __init__.py files are created"
echo "   • Verify conda environment activation"

echo ""
echo "2. Memory Issues:"
echo "   • Reduce batch size in configs"
echo "   • Use CPU if GPU memory insufficient"
echo "   • Monitor memory usage with nvidia-smi"

echo ""
echo "3. Data Loading Issues:"
echo "   • Verify data/synthetic/ contains .pkl and .csv files"
echo "   • Check file permissions and paths"
echo "   • Ensure pickle files are not corrupted"

echo ""
echo "4. Training Issues:"
echo "   • Check learning rates and loss functions"
echo "   • Verify gradient flow (no NaN values)"
echo "   • Monitor training curves for convergence"

echo ""
echo "=============================================="
echo "ADVANCED FEATURES"
echo "=============================================="

echo ""
echo "🧪 Future Enhancements (Week 5-6):"
echo "   • Ablation studies for each component"
echo "   • Hyperparameter optimization"
echo "   • Cross-validation on temporal splits"
echo "   • Real-world Bitcoin data evaluation"

echo ""
echo "📊 Additional Metrics:"
echo "   • Early detection capability (t-k prediction)"
echo "   • Temporal consistency measures"
echo "   • Memory utilization efficiency"
echo "   • Computational complexity analysis"

echo ""
echo "🎨 Enhanced Visualizations:"
echo "   • Interactive temporal network plots"
echo "   • Memory attention heatmaps"
echo "   • Component contribution analysis"
echo "   • Real-time anomaly monitoring dashboard"

echo ""
echo "=============================================="
echo "Setup completed! Ready for temporal experiment."
echo "=============================================="
