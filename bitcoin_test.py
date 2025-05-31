#!/usr/bin/env python3
"""
Quick Bitcoin Analysis Test
Tests Bitcoin fraud detection before full comprehensive suite
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from temporal_anomaly_detector import TemporalAnomalyDetector
import os

def test_bitcoin_fraud_detection():
    """Test Bitcoin fraud detection capabilities"""
    print("💰 BITCOIN FRAUD DETECTION TEST")
    print("="*50)
    
    # Load Bitcoin Alpha (smaller network for testing)
    filepath = 'data/processed/bitcoin_alpha_processed.csv'
    
    if not os.path.exists(filepath):
        print(f"❌ Bitcoin data not found: {filepath}")
        return
    
    print(f"📥 Loading Bitcoin Alpha network...")
    df = pd.read_csv(filepath)
    
    print(f"📊 Network Statistics:")
    print(f"   Total edges: {len(df):,}")
    print(f"   Unique nodes: {len(set(df['source'].tolist() + df['target'].tolist())):,}")
    print(f"   Negative ratings: {len(df[df['rating'] < 0]):,} ({len(df[df['rating'] < 0])/len(df)*100:.1f}%)")
    
    # Create simplified temporal analysis
    print(f"\n🔍 Creating temporal analysis...")
    
    # Sort by timestamp and take recent data (for speed)
    df_sorted = df.sort_values('timestamp')
    recent_data = df_sorted.tail(1000)  # Last 1000 interactions
    
    # Get unique nodes in recent data
    recent_nodes = list(set(recent_data['source_idx'].tolist() + recent_data['target_idx'].tolist()))
    num_nodes = len(recent_nodes)
    
    print(f"   Analyzing recent {len(recent_data)} interactions")
    print(f"   Involving {num_nodes} users")
    
    # Initialize detector
    detector = TemporalAnomalyDetector(
        num_nodes=num_nodes,
        node_feature_dim=8,
        hidden_dim=64,
        embedding_dim=32
    )
    
    print(f"✅ Detector initialized ({sum(p.numel() for p in detector.get_temporal_parameters()):,} parameters)")
    
    # Create temporal windows
    print(f"\n🕒 Creating temporal windows...")
    
    # Split into time windows
    recent_data['datetime'] = pd.to_datetime(recent_data['timestamp'], unit='s')
    recent_data['week'] = recent_data['datetime'].dt.to_period('W')
    
    anomaly_scores = []
    fraud_indicators = []
    
    node_mapping = {node: idx for idx, node in enumerate(recent_nodes)}
    
    for week, week_data in recent_data.groupby('week'):
        if len(week_data) < 5:
            continue
            
        print(f"   Processing week {week}: {len(week_data)} edges")
        
        # Create graph for this week
        valid_edges = []
        for _, row in week_data.iterrows():
            src_orig, tgt_orig = row['source_idx'], row['target_idx']
            if src_orig in node_mapping and tgt_orig in node_mapping:
                valid_edges.append([node_mapping[src_orig], node_mapping[tgt_orig]])
        
        if len(valid_edges) < 3:
            continue
            
        edge_index = torch.tensor(valid_edges).t()
        
        # Create node features
        node_features = torch.zeros(num_nodes, 8)
        
        for _, row in week_data.iterrows():
            src_orig, tgt_orig, rating = row['source_idx'], row['target_idx'], row['rating']
            
            if src_orig in node_mapping and tgt_orig in node_mapping:
                src_idx = node_mapping[src_orig]
                tgt_idx = node_mapping[tgt_orig]
                
                # Basic features
                node_features[src_idx, 0] += 1  # Out-degree
                node_features[tgt_idx, 1] += 1  # In-degree
                node_features[tgt_idx, 2] += rating  # Rating sum
                
                if rating < 0:
                    node_features[tgt_idx, 3] += 1  # Negative count
        
        # Normalize features
        for i in range(num_nodes):
            if node_features[i, 1] > 0:  # If has incoming edges
                node_features[i, 4] = node_features[i, 2] / node_features[i, 1]  # Avg rating
                node_features[i, 5] = node_features[i, 3] / node_features[i, 1]  # Negative ratio
        
        # Create graph
        graph = Data(x=node_features, edge_index=edge_index)
        
        # Determine if this week has suspicious activity
        negative_ratio = len(week_data[week_data['rating'] < 0]) / len(week_data)
        is_suspicious = negative_ratio > 0.3  # More than 30% negative
        
        # Process with temporal detector
        try:
            results = detector.temporal_memory.process_graph(
                graph.x, graph.edge_index, float(len(anomaly_scores)), 
                is_normal=not is_suspicious
            )
            
            score = detector.temporal_memory.compute_unified_anomaly_score(results)
            anomaly_scores.append(score.item())
            fraud_indicators.append(is_suspicious)
            
            print(f"     Score: {score.item():.3f}, Suspicious: {is_suspicious}")
            
        except Exception as e:
            print(f"     ❌ Error processing week: {e}")
            continue
    
    # Analyze results
    print(f"\n📊 FRAUD DETECTION RESULTS:")
    print(f"   Processed weeks: {len(anomaly_scores)}")
    
    if len(anomaly_scores) > 0:
        normal_scores = [s for s, is_fraud in zip(anomaly_scores, fraud_indicators) if not is_fraud]
        suspicious_scores = [s for s, is_fraud in zip(anomaly_scores, fraud_indicators) if is_fraud]
        
        if normal_scores and suspicious_scores:
            normal_avg = np.mean(normal_scores)
            suspicious_avg = np.mean(suspicious_scores)
            separation = suspicious_avg / max(normal_avg, 0.001)
            
            print(f"   Normal weeks: {len(normal_scores)}, avg score: {normal_avg:.3f}")
            print(f"   Suspicious weeks: {len(suspicious_scores)}, avg score: {suspicious_avg:.3f}")
            print(f"   Separation ratio: {separation:.2f}x")
            
            if separation > 1.2:
                print("✅ GOOD: Can distinguish suspicious periods!")
            else:
                print("⚠️ WEAK: Limited separation between normal/suspicious")
                
            return {
                'success': True,
                'separation_ratio': separation,
                'normal_scores': normal_scores,
                'suspicious_scores': suspicious_scores,
                'weeks_processed': len(anomaly_scores)
            }
        else:
            print("⚠️ Need both normal and suspicious periods for comparison")
    else:
        print("❌ No weeks processed successfully")
    
    return {'success': False}

def main():
    """Run Bitcoin fraud detection test"""
    try:
        results = test_bitcoin_fraud_detection()
        
        if results.get('success', False):
            print(f"\n🎉 BITCOIN TEST SUCCESSFUL!")
            print(f"Ready to run full comprehensive testing suite.")
            print(f"Next: python comprehensive_testing.py")
        else:
            print(f"\n⚠️ Bitcoin test had issues - check above for details")
            
    except Exception as e:
        print(f"❌ Bitcoin test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
