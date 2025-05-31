#!/usr/bin/env python3
"""
Quick check of TemporalAnomalyMemory constructor parameters
"""

import inspect
from temporal_memory_module import TemporalAnomalyMemory

def check_constructor():
    """Check the actual constructor signature"""
    print("🔍 Checking TemporalAnomalyMemory constructor...")
    
    # Get the constructor signature
    signature = inspect.signature(TemporalAnomalyMemory.__init__)
    print("Constructor signature:")
    print(signature)
    
    # Get parameter details
    print("\nParameters:")
    for name, param in signature.parameters.items():
        if name != 'self':
            print(f"  {name}: {param}")
    
    # Try to create instance with minimal parameters
    print("\n🧪 Testing minimal initialization...")
    try:
        # Try the most likely combination
        memory = TemporalAnomalyMemory(
            num_nodes=10,
            memory_dim=8
        )
        print("✅ Minimal initialization successful")
        return ['num_nodes', 'memory_dim']
        
    except Exception as e:
        print(f"❌ Minimal failed: {e}")
        
        # Try with additional parameters
        try:
            memory = TemporalAnomalyMemory(
                num_nodes=10,
                memory_dim=8,
                node_feature_dim=4
            )
            print("✅ With node_feature_dim successful")
            return ['num_nodes', 'memory_dim', 'node_feature_dim']
            
        except Exception as e2:
            print(f"❌ With node_feature_dim failed: {e2}")
            
            # Try looking at the source
            try:
                source = inspect.getsource(TemporalAnomalyMemory.__init__)
                print("\nConstructor source:")
                print(source)
            except:
                print("Could not get source code")
            
            return None

if __name__ == "__main__":
    required_params = check_constructor()
    if required_params:
        print(f"\n✅ Required parameters: {required_params}")
    else:
        print("\n❌ Could not determine required parameters")
