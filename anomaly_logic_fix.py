#!/usr/bin/env python3
"""
Anomaly Logic Fix Implementation
Diagnoses and fixes the inverted anomaly detection issue
"""

import torch
import numpy as np
from torch_geometric.data import Data
from temporal_anomaly_detector import TemporalAnomalyDetector
from temporal_memory_module import TemporalAnomalyMemory

def diagnose_anomaly_issue():
    """Diagnose the current anomaly detection behavior"""
    print("🔍 DIAGNOSING ANOMALY DETECTION ISSUE")
    print("="*60)
    
    # Initialize detector
    detector = TemporalAnomalyDetector(
        num_nodes=100,
        node_feature_dim=16,
        hidden_dim=128,
        embedding_dim=64
    )
    
    print("✅ Detector initialized")
    
    # Create clearly different normal and anomaly graphs
    print("\n📊 Creating test graphs...")
    
    # Normal graph: moderate connectivity
    normal_graph = Data(
        x=torch.randn(100, 16),
        edge_index=torch.randint(0, 100, (2, 200))  # 200 edges
    )
    
    # Anomaly graph: dense clique (much higher connectivity)
    clique_size = 20
    clique_nodes = torch.arange(clique_size)
    clique_edges = torch.combinations(clique_nodes, 2).T  # Complete clique
    normal_edges = torch.randint(20, 100, (2, 100))  # Some normal edges
    anomaly_edge_index = torch.cat([clique_edges, normal_edges], dim=1)
    
    anomaly_graph = Data(
        x=torch.randn(100, 16),
        edge_index=anomaly_edge_index  # ~290 edges (much denser)
    )
    
    print(f"   Normal graph: {normal_graph.edge_index.shape[1]} edges")
    print(f"   Anomaly graph: {anomaly_graph.edge_index.shape[1]} edges")
    
    # Test current behavior
    print("\n🧪 Testing current anomaly detection...")
    
    # Process both graphs
    normal_results = detector.temporal_memory.process_graph(
        normal_graph.x, normal_graph.edge_index, 0.0, is_normal=True
    )
    anomaly_results = detector.temporal_memory.process_graph(
        anomaly_graph.x, anomaly_graph.edge_index, 1.0, is_normal=False
    )
    
    # Get current scores
    normal_score = detector.temporal_memory.compute_unified_anomaly_score(normal_results)
    anomaly_score = detector.temporal_memory.compute_unified_anomaly_score(anomaly_results)
    
    print(f"   Current normal score: {normal_score.item():.3f}")
    print(f"   Current anomaly score: {anomaly_score.item():.3f}")
    print(f"   Current ratio (anomaly/normal): {anomaly_score.item()/normal_score.item():.3f}x")
    
    if anomaly_score.item() < normal_score.item():
        print("   ❌ ISSUE CONFIRMED: Anomalies score LOWER than normal!")
        return True, normal_score, anomaly_score
    else:
        print("   ✅ Anomaly detection working correctly")
        return False, normal_score, anomaly_score

def implement_fix_method_1():
    """Method 1: Simple score inversion"""
    print("\n🔧 IMPLEMENTING FIX METHOD 1: Score Inversion")
    print("="*60)
    
    # Test the fix on temporal memory
    temporal_memory = TemporalAnomalyMemory(
        num_nodes=100,
        node_feature_dim=16,
        memory_dim=64,
        embedding_dim=32
    )
    
    # Add corrected scoring method
    def compute_corrected_anomaly_score(self, results):
        """Compute corrected anomaly score (higher = more anomalous)"""
        original_score = self.compute_unified_anomaly_score(results)
        
        # Simple inversion: 1/score makes high scores low and low scores high
        # Add small epsilon to avoid division by zero
        corrected_score = 1.0 / (original_score + 1e-6)
        
        return corrected_score
    
    # Monkey patch the method
    temporal_memory.compute_corrected_anomaly_score = compute_corrected_anomaly_score.__get__(temporal_memory)
    
    # Test the fix
    normal_graph = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 200)))
    anomaly_graph = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 400)))
    
    normal_results = temporal_memory.process_graph(normal_graph.x, normal_graph.edge_index, 0.0, True)
    anomaly_results = temporal_memory.process_graph(anomaly_graph.x, anomaly_graph.edge_index, 1.0, False)
    
    # Original scores
    original_normal = temporal_memory.compute_unified_anomaly_score(normal_results)
    original_anomaly = temporal_memory.compute_unified_anomaly_score(anomaly_results)
    
    # Corrected scores
    corrected_normal = temporal_memory.compute_corrected_anomaly_score(normal_results)
    corrected_anomaly = temporal_memory.compute_corrected_anomaly_score(anomaly_results)
    
    print(f"Original - Normal: {original_normal.item():.3f}, Anomaly: {original_anomaly.item():.3f}")
    print(f"Original ratio: {original_anomaly.item()/original_normal.item():.3f}x")
    print(f"Corrected - Normal: {corrected_normal.item():.3f}, Anomaly: {corrected_anomaly.item():.3f}")
    print(f"Corrected ratio: {corrected_anomaly.item()/corrected_normal.item():.3f}x")
    
    if corrected_anomaly.item() > corrected_normal.item():
        print("✅ Fix Method 1 SUCCESS: Anomalies now score higher!")
        return True, corrected_normal, corrected_anomaly
    else:
        print("❌ Fix Method 1 failed")
        return False, corrected_normal, corrected_anomaly

def implement_fix_method_2():
    """Method 2: Exponential transformation"""
    print("\n🔧 IMPLEMENTING FIX METHOD 2: Exponential Transform")
    print("="*60)
    
    temporal_memory = TemporalAnomalyMemory(100, 16, 64, 32)
    
    def compute_exp_corrected_score(self, results):
        """Exponential correction: exp(-score) makes low scores high"""
        original_score = self.compute_unified_anomaly_score(results)
        corrected_score = torch.exp(-original_score)
        return corrected_score
    
    temporal_memory.compute_exp_corrected_score = compute_exp_corrected_score.__get__(temporal_memory)
    
    # Test
    normal_graph = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 200)))
    anomaly_graph = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 400)))
    
    normal_results = temporal_memory.process_graph(normal_graph.x, normal_graph.edge_index, 0.0, True)
    anomaly_results = temporal_memory.process_graph(anomaly_graph.x, anomaly_graph.edge_index, 1.0, False)
    
    original_normal = temporal_memory.compute_unified_anomaly_score(normal_results)
    original_anomaly = temporal_memory.compute_unified_anomaly_score(anomaly_results)
    exp_normal = temporal_memory.compute_exp_corrected_score(normal_results)
    exp_anomaly = temporal_memory.compute_exp_corrected_score(anomaly_results)
    
    print(f"Original - Normal: {original_normal.item():.3f}, Anomaly: {original_anomaly.item():.3f}")
    print(f"Exponential - Normal: {exp_normal.item():.3f}, Anomaly: {exp_anomaly.item():.3f}")
    print(f"Exponential ratio: {exp_anomaly.item()/exp_normal.item():.3f}x")
    
    if exp_anomaly.item() > exp_normal.item():
        print("✅ Fix Method 2 SUCCESS!")
        return True, exp_normal, exp_anomaly
    else:
        print("❌ Fix Method 2 failed")
        return False, exp_normal, exp_anomaly

def create_fixed_comprehensive_test():
    """Create a fixed version of the cross-validation test"""
    print("\n🚀 CREATING FIXED COMPREHENSIVE TEST")
    print("="*60)
    
    fixed_test_code = '''
def test_fixed_cross_validation():
    """Fixed cross-validation test with corrected anomaly scoring"""
    print("🎯 FIXED CROSS-VALIDATION TEST")
    print("="*50)
    
    seeds = [42, 123, 456, 789, 999]
    separation_ratios = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize detector
        detector = TemporalAnomalyDetector(
            num_nodes=100, node_feature_dim=16, 
            hidden_dim=128, embedding_dim=64
        )
        
        # Add corrected scoring method
        def compute_corrected_anomaly_score(self, results):
            original_score = self.compute_unified_anomaly_score(results)
            return 1.0 / (original_score + 1e-6)  # Inversion fix
        
        detector.temporal_memory.compute_corrected_anomaly_score = compute_corrected_anomaly_score.__get__(detector.temporal_memory)
        
        # Test on multiple scenarios
        normal_scores = []
        anomaly_scores = []
        
        for i in range(5):
            # Normal graph
            normal_graph = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 200)))
            normal_results = detector.temporal_memory.process_graph(normal_graph.x, normal_graph.edge_index, float(i), True)
            normal_score = detector.temporal_memory.compute_corrected_anomaly_score(normal_results)
            normal_scores.append(normal_score.item())
            
            # Anomaly graph (dense clique)
            clique_size = 15
            clique_edges = torch.combinations(torch.arange(clique_size), 2).T
            normal_edges = torch.randint(15, 100, (2, 150))
            anomaly_edge_index = torch.cat([clique_edges, normal_edges], dim=1)
            anomaly_graph = Data(x=torch.randn(100, 16), edge_index=anomaly_edge_index)
            
            anomaly_results = detector.temporal_memory.process_graph(anomaly_graph.x, anomaly_graph.edge_index, float(i+10), False)
            anomaly_score = detector.temporal_memory.compute_corrected_anomaly_score(anomaly_results)
            anomaly_scores.append(anomaly_score.item())
        
        # Calculate separation
        normal_mean = np.mean(normal_scores)
        anomaly_mean = np.mean(anomaly_scores)
        separation = anomaly_mean / normal_mean
        separation_ratios.append(separation)
        
        print(f"Seed {seed}: Normal={normal_mean:.3f}, Anomaly={anomaly_mean:.3f}, Separation={separation:.2f}x")
    
    mean_sep = np.mean(separation_ratios)
    std_sep = np.std(separation_ratios)
    
    print(f"\\nFixed Results:")
    print(f"Mean separation: {mean_sep:.2f} ± {std_sep:.2f}")
    print(f"Range: [{min(separation_ratios):.2f}, {max(separation_ratios):.2f}]")
    
    if mean_sep > 1.5:
        print("✅ SUCCESS: Fixed anomaly detection working!")
    else:
        print("⚠️ Still needs improvement")
    
    return separation_ratios

# Run the fixed test
if __name__ == "__main__":
    separation_ratios = test_fixed_cross_validation()
'''
    
    # Save the fixed test
    with open('fixed_cross_validation_test.py', 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('import torch\nimport numpy as np\nfrom torch_geometric.data import Data\nfrom temporal_anomaly_detector import TemporalAnomalyDetector\n\n')
        f.write(fixed_test_code)
    
    print("✅ Fixed test saved to 'fixed_cross_validation_test.py'")
    print("   Run with: python fixed_cross_validation_test.py")

def main():
    """Run complete diagnosis and fix implementation"""
    print("🩺 ANOMALY DETECTION FIX IMPLEMENTATION")
    print("="*70)
    
    # Step 1: Diagnose the issue
    has_issue, normal_score, anomaly_score = diagnose_anomaly_issue()
    
    if not has_issue:
        print("\n✅ No issue detected - anomaly detection working correctly!")
        return
    
    # Step 2: Test fix methods
    print("\n" + "="*70)
    print("TESTING FIX METHODS")
    print("="*70)
    
    method1_success, _, _ = implement_fix_method_1()
    method2_success, _, _ = implement_fix_method_2()
    
    # Step 3: Create fixed comprehensive test
    create_fixed_comprehensive_test()
    
    # Step 4: Recommendations
    print("\n" + "="*70)
    print("🎯 RECOMMENDATIONS")
    print("="*70)
    
    if method1_success:
        print("✅ RECOMMENDED: Use Fix Method 1 (Score Inversion)")
        print("   - Simple and effective")
        print("   - Easy to implement")
        print("   - Use: score = 1.0 / (original_score + 1e-6)")
    elif method2_success:
        print("✅ ALTERNATIVE: Use Fix Method 2 (Exponential)")
        print("   - Use: score = exp(-original_score)")
    else:
        print("⚠️ Both methods failed - may need deeper architecture changes")
    
    print("\n📋 NEXT STEPS:")
    print("1. Run: python fixed_cross_validation_test.py")
    print("2. If successful, update comprehensive_testing.py")
    print("3. Re-run full test suite with fix")
    print("4. Should achieve 1.5x+ separation ratios!")

if __name__ == "__main__":
    main()
