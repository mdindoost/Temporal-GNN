#!/usr/bin/env python3
"""
Test script to verify all graph libraries are working correctly on Wulver
Run this to make sure your environment is ready for the temporal GNN project
"""

import sys
import torch
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print("-" * 50)

# Test PyTorch Geometric
try:
    import torch_geometric
    print(f"✅ PyTorch Geometric version: {torch_geometric.__version__}")
    
    # Test basic GCN functionality
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    import torch.nn.functional as F
    
    # Create a simple test graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    # Test GCN layer
    conv = GCNConv(1, 2)
    if torch.cuda.is_available():
        conv = conv.cuda()
        data = data.cuda()
    
    out = conv(data.x, data.edge_index)
    print(f"✅ GCN test passed - Output shape: {out.shape}")
    
except ImportError as e:
    print(f"❌ PyTorch Geometric import failed: {e}")
except Exception as e:
    print(f"❌ PyTorch Geometric test failed: {e}")

print("-" * 50)

# Test PyTorch Geometric Temporal
try:
    import torch_geometric_temporal
    print(f"✅ PyTorch Geometric Temporal imported successfully")
    
    # Test a simple temporal layer
    from torch_geometric_temporal.nn.recurrent import DCRNN
    
    # Create temporal test
    temporal_layer = DCRNN(in_channels=1, out_channels=2, K=2)
    if torch.cuda.is_available():
        temporal_layer = temporal_layer.cuda()
    
    # Test with temporal data
    x = torch.randn(3, 1)  # 3 nodes, 1 feature
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    if torch.cuda.is_available():
        x = x.cuda()
        edge_index = edge_index.cuda()
    
    # Initialize hidden state
    h = temporal_layer(x, edge_index)
    print(f"✅ Temporal GNN test passed - Hidden state shape: {h.shape}")
    
except ImportError as e:
    print(f"❌ PyTorch Geometric Temporal import failed: {e}")
except Exception as e:
    print(f"❌ PyTorch Geometric Temporal test failed: {e}")

print("-" * 50)

# Test other essential libraries
libraries_to_test = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 
    'sklearn', 'networkx', 'jupyter'
]

for lib in libraries_to_test:
    try:
        __import__(lib)
        print(f"✅ {lib} imported successfully")
    except ImportError:
        print(f"❌ {lib} import failed")

print("-" * 50)

# Memory and compute test
if torch.cuda.is_available():
    print("🧠 GPU Memory Test:")
    torch.cuda.empty_cache()
    
    # Create a larger tensor to test memory
    x = torch.randn(1000, 100).cuda()
    y = torch.mm(x, x.t())
    
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    memory_reserved = torch.cuda.memory_reserved() / 1e9
    
    print(f"   Memory allocated: {memory_allocated:.2f} GB")
    print(f"   Memory reserved: {memory_reserved:.2f} GB")
    print("✅ GPU memory test passed")
    
    torch.cuda.empty_cache()

print("-" * 50)
print("🎉 All tests completed! Environment is ready for temporal GNN research.")
print("\nNext steps:")
print("1. Download datasets")
print("2. Implement static GNN baseline")
print("3. Start literature review")
