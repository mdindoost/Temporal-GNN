#!/usr/bin/env python3
"""
Simplified Temporal Experiment - Focus on Anomaly Detection Results
Avoids gradient training issues while demonstrating temporal capabilities
"""

import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from temporal_memory_module import TemporalAnomalyMemory
from static_gnn_baseline import StaticAnomalyDetector, evaluate_anomaly_detection

def load_synthetic_data(data_path: str = 'data/synthetic/'):
    """Load synthetic temporal graph data"""
    print("Loading synthetic temporal graph data...")
    
    pkl_file = os.path.join(data_path, 'temporal_graph_with_anomalies.pkl')
    
    with open(pkl_file, 'rb') as f:
        temporal_data = pickle.load(f)
    
    print(f"Loaded {len(temporal_data)} timestamps")
    known_anomalies = [15, 30, 45]
    return temporal_data, known_anomalies

def dict_to_tensors(graph_dict: Dict, feature_dim: int = 16):
    """Convert graph dictionary to tensors"""
    num_nodes = graph_dict['num_nodes']
    edges = graph_dict['edges']
    
    if num_nodes == 0:
        return torch.empty((0, feature_dim)), torch.empty((2, 0), dtype=torch.long)
    
    # Create consistent features based on graph structure
    np.random.seed(hash(str(graph_dict['timestamp'])) % 2**32)  # Consistent per timestamp
    features = torch.randn(num_nodes, feature_dim) * 0.1
    
    # Create edge index with validation
    if len(edges) > 0:
        valid_edges = [(u, v) for u, v in edges if u < num_nodes and v < num_nodes]
        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return features, edge_index

def run_simplified_temporal_experiment():
    """Run temporal experiment without gradient training"""
    print("="*80)
    print("SIMPLIFIED TEMPORAL ANOMALY DETECTION EXPERIMENT")
    print("="*80)
    
    # Load data
    temporal_data, known_anomalies = load_synthetic_data()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    # Initialize temporal memory system
    temporal_memory = TemporalAnomalyMemory(100, 16, 64, 32)
    
    # Move to device
    temporal_memory.node_memory = temporal_memory.node_memory.to(device)
    temporal_memory.graph_memory = temporal_memory.graph_memory.to(device)
    temporal_memory.temporal_encoder = temporal_memory.temporal_encoder.to(device)
    temporal_memory.trajectory_predictor = temporal_memory.trajectory_predictor.to(device)
    
    print(f"Temporal memory initialized on {device}")
    
    # Process all graphs and collect results
    print("\nProcessing temporal graphs...")
    results = []
    temporal_scores = []
    
    # Phase 1: Process normal graphs to build memory
    print("Phase 1: Building temporal memory from normal graphs...")
    for graph_dict in temporal_data:
        if not graph_dict['is_anomaly'] and graph_dict['num_nodes'] > 0:
            features, edge_index = dict_to_tensors(graph_dict)
            features, edge_index = features.to(device), edge_index.to(device)
            
            with torch.no_grad():
                graph_results = temporal_memory.process_graph(
                    features, edge_index, float(graph_dict['timestamp']), is_normal=True
                )
    
    print("✅ Temporal memory trained on normal patterns")
    
    # Phase 2: Evaluate all graphs (including anomalies)
    print("\nPhase 2: Evaluating all graphs with temporal memory...")
    temporal_memory.is_training_phase = False  # Switch to evaluation mode
    
    for graph_dict in temporal_data:
        if graph_dict['num_nodes'] == 0:
            continue
            
        features, edge_index = dict_to_tensors(graph_dict)
        features, edge_index = features.to(device), edge_index.to(device)
        
        with torch.no_grad():
            # Temporal approach
            graph_results = temporal_memory.process_graph(
                features, edge_index, float(graph_dict['timestamp']), is_normal=False
            )
            temporal_score = temporal_memory.compute_unified_anomaly_score(graph_results)
            
        temporal_scores.append(temporal_score.item())
        
        results.append({
            'timestamp': graph_dict['timestamp'],
            'is_anomaly': graph_dict['is_anomaly'],
            'anomaly_type': graph_dict['anomaly_type'],
            'temporal_score': temporal_score.item(),
            'num_nodes': graph_dict['num_nodes'],
            'num_edges': graph_dict['num_edges']
        })
        
        # Visual indicator
        anomaly_symbol = "🚨" if graph_dict['is_anomaly'] else "✅"
        print(f"T={graph_dict['timestamp']:2d}: {anomaly_symbol} {graph_dict['anomaly_type']:12s} | "
              f"Score={temporal_score.item():.4f} | Nodes={graph_dict['num_nodes']:3d}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Analyze performance
    print("\n" + "="*60)
    print("TEMPORAL ANOMALY DETECTION ANALYSIS")
    print("="*60)
    
    # Separate normal and anomaly scores
    normal_scores = results_df[~results_df['is_anomaly']]['temporal_score'].values
    anomaly_scores = results_df[results_df['is_anomaly']]['temporal_score'].values
    
    # Compute statistics
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        normal_mean = np.mean(normal_scores)
        normal_std = np.std(normal_scores)
        anomaly_mean = np.mean(anomaly_scores)
        anomaly_std = np.std(anomaly_scores)
        
        print(f"Normal graphs:")
        print(f"  Count: {len(normal_scores)}")
        print(f"  Mean score: {normal_mean:.4f} ± {normal_std:.4f}")
        print(f"  Range: [{np.min(normal_scores):.4f}, {np.max(normal_scores):.4f}]")
        
        print(f"\nAnomalous graphs:")
        print(f"  Count: {len(anomaly_scores)}")
        print(f"  Mean score: {anomaly_mean:.4f} ± {anomaly_std:.4f}")
        print(f"  Range: [{np.min(anomaly_scores):.4f}, {np.max(anomaly_scores):.4f}]")
        
        # Compute separation metrics
        separation_ratio = anomaly_mean / normal_mean if normal_mean > 0 else 0
        print(f"\nSeparation Analysis:")
        print(f"  Ratio (anomaly/normal): {separation_ratio:.2f}x")
        
        if separation_ratio > 1.1:
            print("  ✅ Good separation - anomalies have higher scores")
        elif separation_ratio > 0.9:
            print("  ⚠️  Moderate separation - needs parameter tuning")
        else:
            print("  ❌ Poor separation - significant tuning needed")
        
        # Evaluate detection performance
        labels = results_df['is_anomaly'].astype(int).values
        scores = results_df['temporal_score'].values
        
        metrics = evaluate_anomaly_detection(scores, labels)
        
        print(f"\nDetection Performance:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Average Precision: {metrics['ap']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Compare with random baseline
        if metrics['auc'] > 0.6:
            print("  ✅ Performance above random baseline")
        else:
            print("  ⚠️  Performance below expectations")
    
    # Analyze specific anomalies
    print(f"\nSpecific Anomaly Analysis:")
    for anomaly_time in known_anomalies:
        anomaly_row = results_df[results_df['timestamp'] == anomaly_time]
        if not anomaly_row.empty:
            score = anomaly_row['temporal_score'].iloc[0]
            anomaly_type = anomaly_row['anomaly_type'].iloc[0]
            print(f"  T={anomaly_time} ({anomaly_type}): {score:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/simplified_temporal_results.csv', index=False)
    
    # Create visualization
    create_temporal_visualization(results_df, known_anomalies)
    
    print(f"\n📊 Results saved to 'results/simplified_temporal_results.csv'")
    print("🎉 Simplified temporal experiment completed successfully!")
    
    return results_df

def create_temporal_visualization(results_df, known_anomalies):
    """Create visualization of temporal results"""
    plt.figure(figsize=(12, 6))
    
    # Plot temporal scores
    plt.plot(results_df['timestamp'], results_df['temporal_score'], 
             'b-', linewidth=2, marker='o', markersize=4, label='Temporal Anomaly Score')
    
    # Mark known anomalies
    for t in known_anomalies:
        anomaly_row = results_df[results_df['timestamp'] == t]
        if not anomaly_row.empty:
            score = anomaly_row['temporal_score'].iloc[0]
            anomaly_type = anomaly_row['anomaly_type'].iloc[0]
            plt.axvline(x=t, color='red', linestyle=':', alpha=0.7, linewidth=2)
            plt.plot(t, score, 'ro', markersize=10, 
                    label=f'Known Anomaly' if t == known_anomalies[0] else "")
            plt.annotate(f'{anomaly_type}', (t, score), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.title('Temporal Anomaly Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/temporal_anomaly_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Visualization saved to 'results/temporal_anomaly_timeline.png'")

if __name__ == "__main__":
    run_simplified_temporal_experiment()
