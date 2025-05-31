#!/usr/bin/env python3
"""
Simple Training Verification Test
Tests if your training pipeline actually works with multiple steps
"""

import torch
import torch.optim as optim
from torch_geometric.data import Data
from temporal_anomaly_detector import TemporalAnomalyDetector
import matplotlib.pyplot as plt

def test_full_training_loop():
    """Test a complete training loop to see if it really works"""
    print("="*60)
    print("FULL TRAINING LOOP TEST")
    print("="*60)
    
    # Initialize detector
    detector = TemporalAnomalyDetector(
        num_nodes=100,
        node_feature_dim=16,
        hidden_dim=64,
        embedding_dim=32
    )
    
    print(f"✅ Detector initialized")
    print(f"📊 Total parameters: {sum(p.numel() for p in detector.get_temporal_parameters()):,}")
    
    # Create synthetic training data
    def create_test_graph(is_anomaly=False):
        if is_anomaly:
            # Create anomalous graph (more edges, different structure)
            node_features = torch.randn(100, 16) * 2.0  # Higher variance
            edge_index = torch.randint(0, 100, (2, 500))  # More edges
        else:
            # Create normal graph
            node_features = torch.randn(100, 16)
            edge_index = torch.randint(0, 100, (2, 200))
        
        return Data(x=node_features, edge_index=edge_index)
    
    # Training parameters
    num_epochs = 10
    learning_rate = 0.001
    optimizer = optim.Adam(detector.get_temporal_parameters(), lr=learning_rate)
    
    # Training history
    losses = []
    normal_scores = []
    anomaly_scores = []
    
    print("\n🚂 Starting training loop...")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Train on normal graphs
        for i in range(5):
            normal_graph = create_test_graph(is_anomaly=False)
            
            # Training step
            optimizer.zero_grad()
            loss = detector.temporal_training_step(normal_graph, float(i), is_normal=True)
            epoch_losses.append(loss)
            
            # Get anomaly score for monitoring
            with torch.no_grad():
                results = detector.temporal_memory.process_graph(
                    normal_graph.x, normal_graph.edge_index, float(i), is_normal=True
                )
                score = detector.temporal_memory.compute_unified_anomaly_score(results)
                normal_scores.append(score.item())
        
        # Train on anomalous graphs
        for i in range(3):
            anomaly_graph = create_test_graph(is_anomaly=True)
            
            # Training step
            optimizer.zero_grad()
            loss = detector.temporal_training_step(anomaly_graph, float(i+10), is_normal=False)
            epoch_losses.append(loss)
            
            # Get anomaly score for monitoring
            with torch.no_grad():
                results = detector.temporal_memory.process_graph(
                    anomaly_graph.x, anomaly_graph.edge_index, float(i+10), is_normal=False
                )
                score = detector.temporal_memory.compute_unified_anomaly_score(results)
                anomaly_scores.append(score.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"📈 Final loss: {losses[-1]:.4f}")
    print(f"📉 Loss change: {losses[0]:.4f} → {losses[-1]:.4f} ({((losses[-1]-losses[0])/losses[0]*100):+.1f}%)")
    
    # Analyze score distributions
    print(f"\n📊 Score Analysis:")
    print(f"Normal graphs   - Mean: {sum(normal_scores)/len(normal_scores):.3f}, Range: [{min(normal_scores):.3f}, {max(normal_scores):.3f}]")
    print(f"Anomaly graphs  - Mean: {sum(anomaly_scores)/len(anomaly_scores):.3f}, Range: [{min(anomaly_scores):.3f}, {max(anomaly_scores):.3f}]")
    
    # Test final performance
    print(f"\n🎯 Final Performance Test:")
    test_normal = create_test_graph(is_anomaly=False)
    test_anomaly = create_test_graph(is_anomaly=True)
    
    with torch.no_grad():
        # Normal graph score
        results_normal = detector.temporal_memory.process_graph(
            test_normal.x, test_normal.edge_index, 100.0, is_normal=True
        )
        score_normal = detector.temporal_memory.compute_unified_anomaly_score(results_normal)
        
        # Anomaly graph score
        results_anomaly = detector.temporal_memory.process_graph(
            test_anomaly.x, test_anomaly.edge_index, 101.0, is_normal=False
        )
        score_anomaly = detector.temporal_memory.compute_unified_anomaly_score(results_anomaly)
    
    print(f"Test normal score:  {score_normal.item():.3f}")
    print(f"Test anomaly score: {score_anomaly.item():.3f}")
    print(f"Separation ratio:   {score_anomaly.item() / max(score_normal.item(), 0.001):.2f}x")
    
    # Save results
    results = {
        'losses': losses,
        'normal_scores': normal_scores,
        'anomaly_scores': anomaly_scores,
        'final_normal_score': score_normal.item(),
        'final_anomaly_score': score_anomaly.item()
    }
    
    return results

def plot_training_results(results):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(results['losses'], 'b-', marker='o')
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot score distributions
    ax2.hist(results['normal_scores'], alpha=0.7, label='Normal', bins=20)
    ax2.hist(results['anomaly_scores'], alpha=0.7, label='Anomaly', bins=20)
    ax2.axvline(results['final_normal_score'], color='blue', linestyle='--', label='Final Normal')
    ax2.axvline(results['final_anomaly_score'], color='red', linestyle='--', label='Final Anomaly')
    ax2.set_title('Anomaly Score Distributions')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_verification_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Results saved to 'training_verification_results.png'")

def main():
    """Run the full training verification test"""
    print("🔧 TRAINING PIPELINE VERIFICATION")
    print("This will test if your training actually works end-to-end")
    
    try:
        results = test_full_training_loop()
        plot_training_results(results)
        
        print("\n" + "="*60)
        print("🎉 TRAINING VERIFICATION COMPLETE!")
        print("="*60)
        
        # Determine if training is working
        if results['losses'][-1] < results['losses'][0]:
            print("✅ TRAINING IS WORKING: Loss decreased over time")
        else:
            print("⚠️  TRAINING CONCERN: Loss did not decrease")
            
        if results['final_anomaly_score'] > results['final_normal_score']:
            print("✅ DETECTION IS WORKING: Anomalies scored higher than normal")
        else:
            print("⚠️  DETECTION CONCERN: Poor anomaly/normal separation")
            
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
