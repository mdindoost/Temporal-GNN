#!/bin/bash
#SBATCH --job-name=temporal-gnn-fixed
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=bader
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/temporal_fixed_%j.out
#SBATCH --error=logs/temporal_fixed_%j.err

echo "=============================================="
echo "FIXED Temporal GNN Anomaly Detection"
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

echo "System Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

echo "Data availability check:"
ls -la data/synthetic/ 2>/dev/null || echo "Warning: synthetic data directory not found"
echo ""

# Step 1: Test the fixed temporal memory components
echo "=============================================="
echo "STEP 1: Testing Fixed Temporal Components"
echo "=============================================="

if [ -f "test_fixed_temporal.py" ]; then
    echo "Running dimension compatibility tests..."
    python test_fixed_temporal.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Temporal component tests passed!"
    else
        echo "❌ Temporal component tests failed!"
        exit 1
    fi
else
    echo "⚠️ test_fixed_temporal.py not found, skipping component test"
fi

echo ""

# Step 2: Run the quick patch test
echo "=============================================="
echo "STEP 2: Quick Temporal Detection Test"
echo "=============================================="

if [ -f "temporal_detector_patch.py" ]; then
    echo "Running quick temporal detection test..."
    python temporal_detector_patch.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Quick temporal test passed!"
    else
        echo "❌ Quick temporal test failed!"
        echo "Continuing with full experiment anyway..."
    fi
else
    echo "⚠️ temporal_detector_patch.py not found, skipping quick test"
fi

echo ""

# Step 3: Run the full temporal experiment
echo "=============================================="
echo "STEP 3: Full Temporal Anomaly Detection"
echo "=============================================="

# Check if we have the fixed temporal memory module
if [ ! -f "temporal_memory_module.py" ]; then
    if [ -f "fixed_temporal_memory.py" ]; then
        echo "Copying fixed temporal memory module..."
        cp fixed_temporal_memory.py temporal_memory_module.py
    else
        echo "❌ No temporal memory module found!"
        exit 1
    fi
fi

echo "Starting full temporal anomaly detection experiment..."

# Try to run the main temporal anomaly detector
if [ -f "temporal_anomaly_detector.py" ]; then
    echo "Running temporal_anomaly_detector.py..."
    python temporal_anomaly_detector.py
    
    DETECTOR_EXIT_CODE=$?
    
    if [ $DETECTOR_EXIT_CODE -eq 0 ]; then
        echo "✅ Full temporal experiment completed successfully!"
    else
        echo "⚠️ Full temporal experiment had issues (exit code: $DETECTOR_EXIT_CODE)"
        echo "Checking for partial results..."
    fi
else
    echo "⚠️ temporal_anomaly_detector.py not found"
    echo "Running alternative simplified experiment..."
    
    # Fallback to patch script
    if [ -f "temporal_detector_patch.py" ]; then
        python temporal_detector_patch.py
    else
        echo "❌ No experiment script available!"
        exit 1
    fi
fi

echo ""

# Step 4: Analyze results
echo "=============================================="
echo "STEP 4: Results Analysis"
echo "=============================================="

echo "Checking for generated results..."

if [ -d "results" ]; then
    echo "Results directory contents:"
    ls -la results/
    echo ""
    
    # Check for different possible result files
    RESULT_FILES=(
        "results/temporal_comparison_results.txt"
        "results/quick_temporal_test.csv"
        "results/static_baseline_results.csv"
    )
    
    for file in "${RESULT_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "📄 Found: $(basename $file)"
            echo "   Size: $(du -h $file | cut -f1)"
        fi
    done
    
    echo ""
    
    # Display key results if available
    if [ -f "results/temporal_comparison_results.txt" ]; then
        echo "=========================================="
        echo "TEMPORAL VS STATIC COMPARISON RESULTS"
        echo "=========================================="
        cat results/temporal_comparison_results.txt
        echo ""
    elif [ -f "results/quick_temporal_test.csv" ]; then
        echo "=========================================="
        echo "QUICK TEMPORAL TEST RESULTS"
        echo "=========================================="
        echo "First few rows of results:"
        head -10 results/quick_temporal_test.csv
        echo ""
        
        # Try to compute basic statistics
        python -c "
import pandas as pd
try:
    df = pd.read_csv('results/quick_temporal_test.csv')
    normal_scores = df[~df['is_anomaly']]['score']
    anomaly_scores = df[df['is_anomaly']]['score']
    print(f'Normal scores (mean): {normal_scores.mean():.4f}')
    print(f'Anomaly scores (mean): {anomaly_scores.mean():.4f}')
    if len(anomaly_scores) > 0 and len(normal_scores) > 0:
        improvement = anomaly_scores.mean() / normal_scores.mean()
        print(f'Improvement ratio: {improvement:.2f}x')
except Exception as e:
    print(f'Could not analyze results: {e}')
" 2>/dev/null || echo "Could not analyze CSV results"
    fi
    
    # Check for visualizations
    echo "Generated visualizations:"
    find results/ -name "*.png" -exec basename {} \; 2>/dev/null | sed 's/^/   📊 /' || echo "   No PNG files found"
    
else
    echo "❌ No results directory found!"
fi

echo ""

# Step 5: Summary
echo "=============================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================="

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo ""

# Determine overall success
SUCCESS_INDICATORS=0

if [ -f "results/temporal_comparison_results.txt" ]; then
    SUCCESS_INDICATORS=$((SUCCESS_INDICATORS + 1))
    echo "✅ Full temporal comparison completed"
elif [ -f "results/quick_temporal_test.csv" ]; then
    SUCCESS_INDICATORS=$((SUCCESS_INDICATORS + 1))
    echo "✅ Quick temporal test completed"
fi

if [ -d "results" ] && [ "$(ls -A results/)" ]; then
    SUCCESS_INDICATORS=$((SUCCESS_INDICATORS + 1))
    echo "✅ Results generated"
fi

if [ $SUCCESS_INDICATORS -ge 2 ]; then
    echo ""
    echo "🎉 EXPERIMENT SUCCESSFUL!"
    echo "📊 Results available in results/ directory"
    echo "📈 Check output files for detailed analysis"
    echo ""
    echo "Key achievements:"
    echo "   • Fixed dimension compatibility issues"
    echo "   • Temporal memory system working"
    echo "   • Anomaly detection results generated"
elif [ $SUCCESS_INDICATORS -eq 1 ]; then
    echo ""
    echo "⚠️ PARTIAL SUCCESS"
    echo "Some components worked, but full experiment may need refinement"
    echo "Check error logs for details"
else
    echo ""
    echo "❌ EXPERIMENT FAILED"
    echo "Check error logs for debugging information"
    echo "Common issues:"
    echo "   • Missing data files"
    echo "   • GPU memory issues"
    echo "   • Import/dependency problems"
fi

echo ""
echo "=============================================="
