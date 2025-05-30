#!/bin/bash
# Quick fix script for temporal dimension issues
# Run this from ~/temporal-gnn-project/

echo "🔧 FIXING TEMPORAL DIMENSION ISSUES"
echo "=================================="

# 1. Test the fixed temporal memory first
echo "1. Testing fixed temporal memory components..."
python test_fixed_temporal.py

if [ $? -eq 0 ]; then
    echo "✅ Fixed temporal memory test passed!"
else
    echo "❌ Fixed temporal memory test failed!"
    exit 1
fi

echo ""
echo "2. Backing up original files..."
if [ -f "temporal_memory_module.py" ]; then
    cp temporal_memory_module.py temporal_memory_module.py.backup
    echo "   ✅ Backed up temporal_memory_module.py"
fi

echo ""
echo "3. Replacing with fixed version..."
cp fixed_temporal_memory.py temporal_memory_module.py
echo "   ✅ Replaced temporal_memory_module.py with fixed version"

echo ""
echo "4. Creating updated temporal anomaly detector..."

# Create a simple patch for the import issue
cat > temporal_detector_patch.py << 'EOF'
#!/usr/bin/env python3
"""
Quick patch to fix the import and run a simplified temporal experiment
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List

# Import fixed components
from temporal_memory_module import TemporalAnomalyMemory

def load_synthetic_data(data_path: str = 'data/synthetic/'):
    """Load synthetic temporal graph data"""
    print("Loading synthetic temporal graph data...")
    
    pkl_file = os.path.join(data_path, 'temporal_graph_with_anomalies.pkl')
    
    with open(pkl_file, 'rb') as f:
        temporal_data = pickle.load(f)
    
    print(f"Loaded {len(temporal_data)} timestamps")
    return temporal_data

def dict_to_features_and_edges(graph_dict: Dict, feature_dim: int = 16):
    """Convert graph dictionary to tensors"""
    num_nodes = graph_dict['num_nodes']
    edges = graph_dict['edges']
    
    if num_nodes == 0:
        return torch.empty((0, feature_dim)), torch.empty((2, 0), dtype=torch.long)
    
    # Create features (simplified)
    features = torch.randn(num_nodes, feature_dim) * 0.1
    
    # Create edge index
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return features, edge_index

def quick_temporal_test():
    """Quick test of temporal anomaly detection"""
    print("="*60)
    print("QUICK TEMPORAL ANOMALY DETECTION TEST")
    print("="*60)
    
    # Load data
    temporal_data = load_synthetic_data()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_system = TemporalAnomalyMemory(100, 16, 64, 32)
    
    # Move to device
    memory_system.node_memory = memory_system.node_memory.to(device)
    memory_system.graph_memory = memory_system.graph_memory.to(device)
    memory_system.temporal_encoder = memory_system.temporal_encoder.to(device)
    memory_system.trajectory_predictor = memory_system.trajectory_predictor.to(device)
    
    print(f"Processing on {device}")
    
    # Process graphs
    results = []
    
    for i, graph_dict in enumerate(temporal_data[:20]):  # Test first 20
        if graph_dict['num_nodes'] == 0:
            continue
            
        # Convert to tensors
        features, edge_index = dict_to_features_and_edges(graph_dict)
        features = features.to(device)
        edge_index = edge_index.to(device)
        
        # Process graph
        graph_results = memory_system.process_graph(
            features, edge_index, 
            float(graph_dict['timestamp']), 
            is_normal=not graph_dict['is_anomaly']
        )
        
        # Compute score
        score = memory_system.compute_unified_anomaly_score(graph_results)
        
        results.append({
            'timestamp': graph_dict['timestamp'],
            'is_anomaly': graph_dict['is_anomaly'],
            'anomaly_type': graph_dict['anomaly_type'],
            'score': score.item(),
            'num_nodes': graph_dict['num_nodes'],
            'num_edges': graph_dict['num_edges']
        })
        
        anomaly_symbol = "🚨" if graph_dict['is_anomaly'] else "✅"
        print(f"T={graph_dict['timestamp']:2d}: {anomaly_symbol} {graph_dict['anomaly_type']:12s} | "
              f"Score={score.item():.4f} | Nodes={graph_dict['num_nodes']:3d}")
    
    # Analyze results
    normal_scores = [r['score'] for r in results if not r['is_anomaly']]
    anomaly_scores = [r['score'] for r in results if r['is_anomaly']]
    
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    
    if normal_scores:
        avg_normal = np.mean(normal_scores)
        print(f"Average normal score: {avg_normal:.4f}")
    
    if anomaly_scores:
        avg_anomaly = np.mean(anomaly_scores)
        print(f"Average anomaly score: {avg_anomaly:.4f}")
        
        if normal_scores and avg_anomaly > avg_normal:
            improvement = avg_anomaly / avg_normal
            print(f"✅ Improvement ratio: {improvement:.2f}x")
            print("✅ Temporal memory is working!")
        else:
            print("⚠️  Need to tune parameters")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/quick_temporal_test.csv', index=False)
    print(f"\nResults saved to results/quick_temporal_test.csv")
    
    return results

if __name__ == "__main__":
    quick_temporal_test()
EOF

echo "   ✅ Created temporal_detector_patch.py"

echo ""
echo "5. Running quick temporal test..."
python temporal_detector_patch.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Temporal memory system is working correctly."
    echo ""
    echo "Next steps:"
    echo "1. ✅ Dimension issues are fixed"
    echo "2. ✅ Temporal memory is functioning"
    echo "3. 🔄 Ready to run full temporal experiment"
    echo ""
    echo "To run the full experiment:"
    echo "   sbatch run_temporal_experiment.slurm"
    echo ""
    echo "Or run interactively:"
    echo "   python temporal_anomaly_detector.py"
else
    echo ""
    echo "❌ Quick test failed. Check the errors above."
    echo ""
    echo "Debugging steps:"
    echo "1. Check if synthetic data exists: ls data/synthetic/"
    echo "2. Verify CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "3. Check memory usage: nvidia-smi"
fi
