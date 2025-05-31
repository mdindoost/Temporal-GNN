#!/usr/bin/env python3
"""
Training Diagnostic Tool
Analyzes the temporal training pipeline to identify exact issues
"""

import torch
import torch.nn.functional as F
import traceback
from temporal_memory_module import TemporalAnomalyMemory
from temporal_anomaly_detector import TemporalAnomalyDetector

def analyze_error_source():
    """Reproduce and analyze the exact training error"""
    print("="*80)
    print("TRAINING ERROR DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Enable anomaly detection for detailed gradient tracking
    torch.autograd.set_detect_anomaly(True)
    
    try:
        # Initialize the detector (same as in your training)
        detector = TemporalAnomalyDetector(
            num_nodes=100,
            node_feature_dim=16, 
            hidden_dim=64,
            embedding_dim=32
        )
        
        print("✅ Detector initialized successfully")
        
        # Create a simple graph for testing
        node_features = torch.randn(100, 16)
        edge_index = torch.randint(0, 100, (2, 300))
        
        from torch_geometric.data import Data
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        print("✅ Test graph created")
        
        # Try the training step that's failing
        print("\n🔍 Testing training step...")
        
        try:
            loss = detector.temporal_training_step(graph_data, 0.0, is_normal=True)
            print(f"✅ Training step successful, loss: {loss}")
            
        except Exception as e:
            print(f"❌ ERROR in training step: {e}")
            print("\nFull stack trace:")
            traceback.print_exc()
            return str(e), traceback.format_exc()
            
    except Exception as e:
        print(f"❌ ERROR in initialization: {e}")
        traceback.print_exc()
        return str(e), traceback.format_exc()
    
    finally:
        torch.autograd.set_detect_anomaly(False)

def analyze_loss_computation():
    """Analyze the loss computation structure"""
    print("\n" + "="*60)
    print("LOSS COMPUTATION ANALYSIS")
    print("="*60)
    
    # Get the source code of the training step
    import inspect
    from temporal_anomaly_detector import TemporalAnomalyDetector
    
    try:
        source = inspect.getsource(TemporalAnomalyDetector.temporal_training_step)
        print("Current temporal_training_step method:")
        print("-" * 40)
        print(source)
        
    except Exception as e:
        print(f"Could not retrieve source: {e}")
        
    # Analyze the unified anomaly scoring
    print("\nUnified anomaly scoring method:")
    print("-" * 40)
    
    try:
        temporal_memory = TemporalAnomalyMemory(10, 8)
        source = inspect.getsource(temporal_memory.compute_unified_anomaly_score)
        print(source)
        
    except Exception as e:
        print(f"Could not retrieve scoring source: {e}")

def analyze_memory_updates():
    """Analyze where in-place operations occur in memory updates"""
    print("\n" + "="*60)
    print("MEMORY UPDATE ANALYSIS")
    print("="*60)
    
    # Check node memory update
    try:
        from temporal_memory_module import NodeMemoryModule
        source = inspect.getsource(NodeMemoryModule.update_memory)
        print("NodeMemoryModule.update_memory:")
        print("-" * 40)
        print(source)
        
    except Exception as e:
        print(f"Could not retrieve node memory source: {e}")
    
    # Check graph memory update  
    try:
        from temporal_memory_module import GraphMemoryModule
        source = inspect.getsource(GraphMemoryModule.update_memory)
        print("\nGraphMemoryModule.update_memory:")
        print("-" * 40)
        print(source)
        
    except Exception as e:
        print(f"Could not retrieve graph memory source: {e}")

def analyze_model_architecture():
    """Analyze the model architecture and component connections"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    # Initialize components
    temporal_memory = TemporalAnomalyMemory(10, 8, 32, 16)
    
    print("Component architecture:")
    print(f"1. Node Memory: {type(temporal_memory.node_memory).__name__}")
    print(f"   - Memory dim: {temporal_memory.node_memory.memory_dim}")
    print(f"   - Message dim: {temporal_memory.node_memory.message_dim}")
    
    print(f"\n2. Graph Memory: {type(temporal_memory.graph_memory).__name__}")
    print(f"   - Memory dim: {temporal_memory.graph_memory.memory_dim}")
    print(f"   - Feature dim: {temporal_memory.graph_memory.graph_feature_dim}")
    
    print(f"\n3. Temporal Encoder: {type(temporal_memory.temporal_encoder).__name__}")
    print(f"   - Output dim: {temporal_memory.temporal_encoder.output_dim}")
    
    print(f"\n4. Trajectory Predictor: {type(temporal_memory.trajectory_predictor).__name__}")
    print(f"   - Embedding dim: {temporal_memory.trajectory_predictor.embedding_dim}")
    
    # Check parameter counts
    total_params = 0
    for name, component in [
        ("node_memory", temporal_memory.node_memory),
        ("graph_memory", temporal_memory.graph_memory), 
        ("temporal_encoder", temporal_memory.temporal_encoder),
        ("trajectory_predictor", temporal_memory.trajectory_predictor)
    ]:
        params = sum(p.numel() for p in component.parameters())
        total_params += params
        print(f"\n{name} parameters: {params:,}")
        
        # Check for buffer parameters (potential in-place operation sources)
        buffers = list(component.named_buffers())
        if buffers:
            print(f"  Buffers: {len(buffers)}")
            for buf_name, buf_tensor in buffers:
                print(f"    {buf_name}: {buf_tensor.shape}")
    
    print(f"\nTotal trainable parameters: {total_params:,}")

def identify_gradient_issues():
    """Identify specific gradient computation issues"""
    print("\n" + "="*60)
    print("GRADIENT ISSUE IDENTIFICATION")
    print("="*60)
    
    # Create a minimal test case
    temporal_memory = TemporalAnomalyMemory(5, 4, 16, 8)
    
    # Test individual components
    print("Testing individual components for gradient issues...")
    
    # Test 1: Node memory
    try:
        node_features = torch.randn(5, 8, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Get memory context
        node_ids = torch.arange(5)
        memory_context = temporal_memory.node_memory.get_memory(node_ids)
        
        # Test encoder
        embeddings = temporal_memory.temporal_encoder(node_features, edge_index, memory_context)
        
        # Test if this can be backpropagated
        loss = torch.mean(embeddings)
        loss.backward()
        
        print("✅ Basic gradient flow works")
        
    except Exception as e:
        print(f"❌ Gradient issue in basic flow: {e}")
        traceback.print_exc()
    
    # Test 2: Memory update with gradients
    try:
        print("\nTesting memory update with gradients...")
        
        # This is likely where the issue occurs
        node_features = torch.randn(5, 4, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        results = temporal_memory.process_graph(
            node_features, edge_index, 0.0, is_normal=True
        )
        
        score = temporal_memory.compute_unified_anomaly_score(results)
        score.backward()
        
        print("✅ Memory update with gradients works")
        
    except Exception as e:
        print(f"❌ Memory update gradient issue: {e}")
        traceback.print_exc()

def create_gradient_safe_training_step():
    """Create a gradient-safe version of the training step"""
    print("\n" + "="*60)
    print("GRADIENT-SAFE TRAINING STEP DESIGN")
    print("="*60)
    
    safe_training_code = '''
def gradient_safe_training_step(self, graph_data, timestamp, is_normal=True):
    """Gradient-safe training step that avoids in-place operations"""
    
    # Step 1: Process graph WITHOUT gradients (memory updates)
    with torch.no_grad():
        # Update temporal memory (no gradients needed for memory)
        _ = self.temporal_memory.process_graph(
            graph_data.x.detach(), 
            graph_data.edge_index.detach(), 
            timestamp, 
            is_normal
        )
    
    # Step 2: Forward pass WITH gradients (for actual training)
    self.optimizer.zero_grad()
    
    # Get current memory state (detached to avoid gradient issues)
    node_ids = torch.arange(min(self.temporal_memory.num_nodes, graph_data.x.shape[0]), 
                           device=graph_data.x.device)
    memory_context = self.temporal_memory.node_memory.get_memory(node_ids).detach()
    
    # Forward pass with gradients enabled
    node_embeddings = self.temporal_memory.temporal_encoder(
        graph_data.x, graph_data.edge_index, memory_context
    )
    
    # Compute loss based on embedding properties
    if is_normal:
        # For normal graphs, encourage compact embeddings
        embedding_norm = torch.mean(torch.norm(node_embeddings, p=2, dim=1))
        target_norm = torch.tensor(1.0, device=node_embeddings.device)
        consistency_loss = F.mse_loss(embedding_norm, target_norm)
    else:
        # For anomalous graphs, encourage larger embeddings
        embedding_norm = torch.mean(torch.norm(node_embeddings, p=2, dim=1))
        target_norm = torch.tensor(2.0, device=node_embeddings.device)
        consistency_loss = F.mse_loss(embedding_norm, target_norm)
    
    # Regularization
    reg_loss = 0.001 * torch.mean(node_embeddings ** 2)
    
    # Total loss
    total_loss = consistency_loss + reg_loss
    
    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.get_temporal_parameters(), 1.0)
    self.optimizer.step()
    
    return total_loss.item()
'''
    
    print("Suggested gradient-safe training step:")
    print(safe_training_code)
    
    # Save to file
    with open('gradient_safe_training_step.py', 'w') as f:
        f.write(safe_training_code)
    
    print("✅ Saved gradient-safe training step to 'gradient_safe_training_step.py'")

def main():
    """Run complete diagnostic analysis"""
    print("TEMPORAL TRAINING DIAGNOSTIC SUITE")
    print("=" * 80)
    
    # 1. Analyze the exact error
    print("\n1. REPRODUCING THE EXACT ERROR:")
    error_msg, stack_trace = analyze_error_source()
    
    # 2. Analyze loss computation
    print("\n2. ANALYZING LOSS COMPUTATION:")
    analyze_loss_computation()
    
    # 3. Analyze memory updates
    print("\n3. ANALYZING MEMORY UPDATES:")
    analyze_memory_updates()
    
    # 4. Analyze model architecture
    print("\n4. ANALYZING MODEL ARCHITECTURE:")
    analyze_model_architecture()
    
    # 5. Identify gradient issues
    print("\n5. IDENTIFYING GRADIENT ISSUES:")
    identify_gradient_issues()
    
    # 6. Create solution
    print("\n6. CREATING GRADIENT-SAFE SOLUTION:")
    create_gradient_safe_training_step()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the error analysis above")
    print("2. Check gradient_safe_training_step.py for the solution")
    print("3. Apply the fix to temporal_anomaly_detector.py")
    print("4. Re-test the training pipeline")

if __name__ == "__main__":
    main()
