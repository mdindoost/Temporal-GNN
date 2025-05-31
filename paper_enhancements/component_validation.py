#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
from temporal_anomaly_detector import TemporalAnomalyDetector


def bitcoin_component_validation():
    """Test individual components on Bitcoin fraud detection"""
    
    from temporal_memory_module import TemporalAnomalyMemory
    
    # Component configurations
    components = {
        'memory_only': {'use_memory': True, 'use_evolution': False, 'use_prediction': False},
        'evolution_only': {'use_memory': False, 'use_evolution': True, 'use_prediction': False},
        'prediction_only': {'use_memory': False, 'use_evolution': False, 'use_prediction': True},
        'full_system': {'use_memory': True, 'use_evolution': True, 'use_prediction': True}
    }
    
    bitcoin_data = pd.read_csv('data/processed/bitcoin_alpha_processed.csv')
    
    component_results = {}
    
    for comp_name, comp_config in components.items():
        print(f"Testing {comp_name} on Bitcoin Alpha...")
        
        # Initialize detector for this component
        detector = TemporalAnomalyDetector(800, 8, 64, 32)  # Bitcoin-sized
        
        # Apply component-specific scoring
        results = test_component_on_bitcoin(detector, bitcoin_data, comp_config)
        
        component_results[comp_name] = {
            'separation_ratio': results['separation'],
            'suspicious_users_detected': results['suspicious_count'],
            'fraud_types_identified': results['fraud_types'],
            'temporal_consistency': results['consistency']
        }
        
        print(f"   {comp_name}: {results['separation']:.2f}x separation")
    
    # Analysis of component strengths
    component_analysis = analyze_component_strengths(component_results)
    
    return component_results, component_analysis

def test_component_on_bitcoin(detector, bitcoin_data, component_config):
    """Test a specific component configuration on Bitcoin data"""
    
    # Create temporal windows
    bitcoin_data['datetime'] = pd.to_datetime(bitcoin_data['timestamp'], unit='s')
    bitcoin_data['month'] = bitcoin_data['datetime'].dt.to_period('M')
    
    monthly_scores = []
    
    for month, month_data in bitcoin_data.groupby('month'):
        if len(month_data) < 20:
            continue
            
        # Convert to graph format
        edge_index = torch.tensor([
            month_data['source_idx'].values,
            month_data['target_idx'].values
        ], dtype=torch.long)
        
        node_features = torch.randn(800, 8)  # Simplified features
        
        # Process with component configuration
        results = detector.temporal_memory.process_graph(
            node_features, edge_index, float(len(monthly_scores)), 
            is_normal=(len(month_data[month_data['rating'] < 0]) / len(month_data) < 0.1)
        )
        
        # Apply component-specific scoring
        if component_config['use_evolution'] and not component_config['use_memory']:
            # Evolution-only scoring
            score = torch.norm(results['node_embeddings'])
        elif component_config['use_memory'] and not component_config['use_evolution']:
            # Memory-only scoring  
            score = torch.mean(results['node_memory_scores'])
        elif component_config['use_prediction']:
            # Prediction-only scoring
            score = torch.var(results['node_embeddings'])
        else:
            # Full system
            score = detector.temporal_memory.compute_unified_anomaly_score(results)
        
        monthly_scores.append({
            'month': month,
            'score': score.item(),
            'negative_ratio': len(month_data[month_data['rating'] < 0]) / len(month_data)
        })
    
    # Calculate component performance
    normal_months = [s for s in monthly_scores if s['negative_ratio'] < 0.1]
    suspicious_months = [s for s in monthly_scores if s['negative_ratio'] > 0.2]
    
    if normal_months and suspicious_months:
        separation = np.mean([s['score'] for s in suspicious_months]) / np.mean([s['score'] for s in normal_months])
    else:
        separation = 1.0
    
    return {
        'separation': separation,
        'suspicious_count': len(suspicious_months),
        'fraud_types': 'temporal_pattern',  # Simplified
        'consistency': np.std([s['score'] for s in monthly_scores])
    }

# Expected runtime: 8-10 hours
# Expected outcome: Component ranking validated on real data
