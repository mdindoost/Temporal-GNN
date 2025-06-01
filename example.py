#!/usr/bin/env python3
"""
Simple example of using TempAnom-GNN for fraud detection
Based on verified paper results
"""

import pandas as pd
import numpy as np
from temporal_anomaly_detector import TemporalAnomalyDetector

def main():
    print("🚀 TempAnom-GNN Example - Verified Configuration")
    print("="*50)
    
    # Load verified dataset
    data_path = 'data/processed/bitcoin_alpha_processed.csv'
    
    try:
        # Initialize with best configuration for early detection
        detector = TemporalAnomalyDetector(
            data_path=data_path,
            alpha=1.0,  # Evolution-only (verified best)
            beta=0.0,
            gamma=0.0
        )
        
        # Train model
        print("Training TempAnom-GNN...")
        detector.train_temporal_model(epochs=50)
        
        # Evaluate
        print("Evaluating on deployment scenarios...")
        early_score = detector.evaluate_early_detection()
        cold_score = detector.evaluate_cold_start()
        
        print(f"✅ Early Detection Score: {early_score:.3f}")
        print(f"✅ Cold Start Score: {cold_score:.3f}")
        print(f"📊 Expected: Early ~0.300, Cold ~0.360 (from Table 5)")
        
    except FileNotFoundError:
        print("❌ Dataset not found. Run 'python download_datasets.py' first")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
