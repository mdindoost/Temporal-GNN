#!/bin/bash
# Final deployment script - fixes all issues and runs the temporal experiment
# Run this from ~/temporal-gnn-project/

echo "🚀 FINAL TEMPORAL MEMORY DEPLOYMENT"
echo "===================================="

# Step 1: Run comprehensive tests
echo "1. Running comprehensive validation tests..."
python comprehensive_test.py

if [ $? -eq 0 ]; then
    echo "✅ All comprehensive tests passed!"
else
    echo "❌ Comprehensive tests failed!"
    echo "Please check the error output above."
    exit 1
fi

echo ""
echo "2. Deploying fixed temporal memory module..."

# Backup original if it exists
if [ -f "temporal_memory_module.py" ]; then
    cp temporal_memory_module.py temporal_memory_module.py.backup.$(date +%Y%m%d_%H%M%S)
    echo "   ✅ Backed up original temporal_memory_module.py"
fi

# Deploy the final fixed version
cp final_fixed_temporal.py temporal_memory_module.py
echo "   ✅ Deployed final_fixed_temporal.py as temporal_memory_module.py"

echo ""
echo "3. Verifying deployment..."

# Quick verification that the module can be imported
python -c "
try:
    from temporal_memory_module import TemporalAnomalyMemory
    print('   ✅ Module import successful')
    
    # Quick instantiation test
    memory_system = TemporalAnomalyMemory(10, 8, 32, 16)
    print('   ✅ Module instantiation successful')
    
except Exception as e:
    print(f'   ❌ Module verification failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Module verification failed!"
    exit 1
fi

echo ""
echo "4. Checking experiment readiness..."

# Check for required files
REQUIRED_FILES=(
    "temporal_memory_module.py"
    "temporal_anomaly_detector.py"
    "static_gnn_baseline.py"
    "data/synthetic/temporal_graph_with_anomalies.pkl"
)

ALL_READY=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ Found: $file"
    else
        echo "   ❌ Missing: $file"
        ALL_READY=false
    fi
done

if [ "$ALL_READY" = false ]; then
    echo ""
    echo "❌ Some required files are missing!"
    echo "Please ensure all necessary files are in place."
    exit 1
fi

echo ""
echo "5. Environment check..."

# Check Python environment
echo "   Python version: $(python --version)"
echo "   PyTorch available: $(python -c 'import torch; print("✅ Yes")' 2>/dev/null || echo "❌ No")"
echo "   PyTorch Geometric available: $(python -c 'import torch_geometric; print("✅ Yes")' 2>/dev/null || echo "❌ No")"
echo "   CUDA available: $(python -c 'import torch; print("✅ Yes" if torch.cuda.is_available() else "❌ No (CPU only)")' 2>/dev/null)"

echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
echo "======================"

echo ""
echo "Choose your next action:"
echo ""
echo "Option 1 - Interactive experiment (recommended for debugging):"
echo "   python temporal_anomaly_detector.py"
echo ""
echo "Option 2 - SLURM batch job (recommended for production):"
echo "   sbatch run_temporal_experiment.slurm"
echo ""
echo "Option 3 - Quick test run:"
echo "   python -c \""
echo "from temporal_memory_module import TemporalAnomalyMemory"
echo "import torch"
echo "print('🎉 Running quick test...')"
echo "memory = TemporalAnomalyMemory(50, 8)"
echo "features = torch.randn(50, 8)"
echo "edges = torch.randint(0, 50, (2, 100))"
echo "results = memory.process_graph(features, edges, 0.0)"
echo "score = memory.compute_unified_anomaly_score(results)"
echo "print(f'✅ Quick test successful! Anomaly score: {score.item():.4f}')"
echo "\""

echo ""
echo "📊 Expected Results:"
echo "   • No dimension errors ✅"
echo "   • No NaN values ✅"
echo "   • No index out of bounds errors ✅"
echo "   • Successful processing of all 50 timestamps ✅"
echo "   • Higher anomaly scores for T=15, 30, 45 ✅"
echo "   • AUC improvement over static baseline (0.33 → >0.8) ✅"

echo ""
echo "🔧 If issues persist:"
echo "   1. Check logs in logs/ directory"
echo "   2. Verify data integrity: ls -la data/synthetic/"
echo "   3. Check GPU memory: nvidia-smi"
echo "   4. Run on CPU if needed: export CUDA_VISIBLE_DEVICES=''"

echo ""
echo "✨ All systems ready for temporal anomaly detection!"
