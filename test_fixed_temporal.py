#!/usr/bin/env python3
"""
Test script for fixed temporal memory components
Run this to verify dimension issues are resolved
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from fixed_temporal_memory import TemporalAnomalyMemory

def test_dimensions():
    """Test all dimension compatibility issues"""
    print("="*60)
    print("TESTING FIXED TEMPORAL MEMORY DIMENSIONS")
    print("="*60)
    
    # Setup
    num_nodes = 100
    node_feature_dim = 16
    memory_dim = 64
    embedding_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Nodes: {num_nodes}")
    print(f"Feature dim: {node_feature_dim}")
    print(f"Memory dim: {memory_dim}")
    print(f"Embedding dim: {embedding_dim}")
    print()
    
    try:
        # Initialize memory system
        print("1. Initializing TemporalAnomalyMemory...")
        memory_system = TemporalAnomalyMemory(num_nodes, node_feature_dim, memory_dim, embedding_dim)
        
        # Move to device
        memory_system.node_memory = memory_system.node_memory.to(device)
        memory_system.graph_memory = memory_system.graph_memory.to(device)
        memory_system.temporal_encoder = memory_system.temporal_encoder.to(device)
        memory_system.trajectory_predictor = memory_system.trajectory_predictor.to(device)
        print("✅ Initialization successful")
        
        # Test with normal graph
        print("\n2. Testing normal graph processing...")
        node_features = torch.randn(num_nodes, node_feature_dim, device=device)
        edge_index = torch.randint(0, num_nodes, (2, 300), device=device)
        
        results = memory_system.process_graph(node_features, edge_index, 0.0, is_normal=True)
        print("✅ Normal graph processing successful")
        
        # Check output dimensions
        print(f"   Node embeddings: {results['node_embeddings'].shape}")
        print(f"   Graph embedding: {results['graph_embedding'].shape}")
        print(f"   Node memory scores: {results['node_memory_scores'].shape}")
        print(f"   Graph memory score: {results['graph_memory_score'].shape}")
        
        # Test anomaly scoring
        print("\n3. Testing anomaly scoring...")
        score = memory_system.compute_unified_anomaly_score(results)
        print(f"✅ Unified anomaly score: {score.item():.4f}")
        
        # Test multiple graphs
        print("\n4. Testing multiple graph sequence...")
        scores = []
        for t in range(5):
            # Normal graph
            node_features = torch.randn(num_nodes, node_feature_dim, device=device)
            edge_count = 300 + torch.randint(-10, 10, (1,)).item()
            edge_index = torch.randint(0, num_nodes, (2, edge_count), device=device)
            
            results = memory_system.process_graph(node_features, edge_index, float(t), is_normal=True)
            score = memory_system.compute_unified_anomaly_score(results)
            scores.append(score.item())
            print(f"   T={t}: Score={score.item():.4f}")
        
        print("✅ Multiple graph processing successful")
        
        # Test anomalous graphs
        print("\n5. Testing anomalous graphs...")
        
        # Star burst
        node_features = torch.randn(num_nodes, node_feature_dim, device=device)
        star_edges = torch.stack([
            torch.zeros(50, device=device, dtype=torch.long),
            torch.randint(1, num_nodes, (50,), device=device)
        ])
        normal_edges = torch.randint(0, num_nodes, (2, 250), device=device)
        edge_index = torch.cat([star_edges, normal_edges], dim=1)
        
        results = memory_system.process_graph(node_features, edge_index, 10.0, is_normal=False)
        star_score = memory_system.compute_unified_anomaly_score(results)
        print(f"   Star burst score: {star_score.item():.4f}")
        
        # Dense clique
        node_features = torch.randn(num_nodes, node_feature_dim, device=device)
        clique_nodes = torch.arange(20, device=device)
        clique_edges = torch.combinations(clique_nodes, 2).t()
        normal_edges = torch.randint(0, num_nodes, (2, 200), device=device)
        edge_index = torch.cat([clique_edges, normal_edges], dim=1)
        
        results = memory_system.process_graph(node_features, edge_index, 11.0, is_normal=False)
        clique_score = memory_system.compute_unified_anomaly_score(results)
        print(f"   Dense clique score: {clique_score.item():.4f}")
        
        print("✅ Anomalous graph processing successful")
        
        # Summary
        print("\n6. Summary of results:")
        avg_normal = sum(scores) / len(scores)
        print(f"   Average normal score: {avg_normal:.4f}")
        print(f"   Star burst score: {star_score.item():.4f}")
        print(f"   Dense clique score: {clique_score.item():.4f}")
        
        if star_score.item() > avg_normal and clique_score.item() > avg_normal:
            print("✅ Anomaly detection working correctly!")
        else:
            print("⚠️  Anomaly detection needs tuning")
        
        print("\n" + "="*60)
        print("ALL DIMENSION TESTS PASSED! ✅")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_data():
    """Test with actual synthetic data format"""
    print("\n" + "="*60)
    print("TESTING WITH SYNTHETIC DATA FORMAT")
    print("="*60)
    
    try:
        # Mock synthetic data format
        graph_dict = {
            'timestamp': 0,
            'edges': [[i, (i+1) % 100] for i in range(150)],  # Simple ring + extra edges
            'node_features': None,
            'num_nodes': 100,
            'num_edges': 150,
            'is_anomaly': False,
            'anomaly_type': 'normal'
        }
        
        print("Testing data conversion...")
        
        # Convert to PyG format (simplified version)
        num_nodes = graph_dict['num_nodes']
        edges = graph_dict['edges']
        feature_dim = 16
        
        # Create features
        features = torch.randn(num_nodes, feature_dim) * 0.1
        
        # Create edge index
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"   Node features shape: {features.shape}")
        print(f"   Edge index shape: {edge_index.shape}")
        
        # Test with temporal memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        memory_system = TemporalAnomalyMemory(num_nodes, feature_dim)
        
        # Move to device
        memory_system.node_memory = memory_system.node_memory.to(device)
        memory_system.graph_memory = memory_system.graph_memory.to(device)
        memory_system.temporal_encoder = memory_system.temporal_encoder.to(device)
        memory_system.trajectory_predictor = memory_system.trajectory_predictor.to(device)
        
        features = features.to(device)
        edge_index = edge_index.to(device)
        
        # Process graph
        results = memory_system.process_graph(
            features, edge_index, float(graph_dict['timestamp']), 
            is_normal=not graph_dict['is_anomaly']
        )
        
        score = memory_system.compute_unified_anomaly_score(results)
        
        print(f"✅ Synthetic data format test successful!")
        print(f"   Anomaly score: {score.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in synthetic data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("TEMPORAL MEMORY DIMENSION FIX VERIFICATION")
    print("=" * 80)
    
    # Test 1: Basic dimension compatibility
    test1_passed = test_dimensions()
    
    # Test 2: Integration with synthetic data format
    test2_passed = test_integration_with_data()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Dimension compatibility test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Synthetic data integration test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The temporal memory system is ready.")
        print("\nNext steps:")
        print("1. Replace temporal_memory_module.py with fixed_temporal_memory.py")
        print("2. Update the imports in temporal_anomaly_detector.py")
        print("3. Re-run the full temporal experiment")
        return True
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
