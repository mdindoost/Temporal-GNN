#!/usr/bin/env python3
"""
Download and prepare datasets for temporal GNN anomaly detection project
This script downloads Bitcoin and synthetic datasets to get started
"""

import os
import requests
import pandas as pd
import numpy as np
import networkx as nx
import torch
from pathlib import Path
import urllib.request
from zipfile import ZipFile

def create_directories():
    """Create necessary data directories"""
    dirs = [
        'data/bitcoin',
        'data/synthetic', 
        'data/reddit',
        'data/processed'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Created data directories")

def download_bitcoin_data():
    """Download Bitcoin Alpha and OTC datasets"""
    print("📥 Downloading Bitcoin datasets...")
    
    # Bitcoin Alpha (trust network)
    alpha_url = "https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.csv.gz"
    alpha_path = "data/bitcoin/soc-sign-bitcoin-alpha.csv.gz"
    
    # Bitcoin OTC (trust network)  
    otc_url = "https://snap.stanford.edu/data/soc-sign-bitcoin-otc.csv.gz"
    otc_path = "data/bitcoin/soc-sign-bitcoin-otc.csv.gz"
    
    try:
        if not os.path.exists(alpha_path):
            urllib.request.urlretrieve(alpha_url, alpha_path)
            print(f"✅ Downloaded Bitcoin Alpha to {alpha_path}")
        else:
            print(f"📂 Bitcoin Alpha already exists at {alpha_path}")
            
        if not os.path.exists(otc_path):
            urllib.request.urlretrieve(otc_url, otc_path)
            print(f"✅ Downloaded Bitcoin OTC to {otc_path}")
        else:
            print(f"📂 Bitcoin OTC already exists at {otc_path}")
            
    except Exception as e:
        print(f"❌ Error downloading Bitcoin data: {e}")
        print("💡 You can manually download from:")
        print(f"   - {alpha_url}")
        print(f"   - {otc_url}")

def create_synthetic_temporal_graph():
    """Create a synthetic temporal graph with known anomalies"""
    print("🎯 Creating synthetic temporal graph...")
    
    np.random.seed(42)
    num_nodes = 100
    num_timestamps = 50
    
    # Base graph structure (small-world network)
    G_base = nx.watts_strogatz_graph(num_nodes, 6, 0.3)
    base_edges = list(G_base.edges())
    
    temporal_data = []
    
    for t in range(num_timestamps):
        # Normal evolution: slight random changes
        current_edges = base_edges.copy()
        
        # Add some temporal variation (10% edge changes)
        num_changes = int(0.1 * len(current_edges))
        
        # Remove some edges
        edges_to_remove = np.random.choice(len(current_edges), num_changes//2, replace=False)
        for idx in sorted(edges_to_remove, reverse=True):
            current_edges.pop(idx)
        
        # Add some new edges
        for _ in range(num_changes//2):
            u, v = np.random.choice(num_nodes, 2, replace=False)
            if (u, v) not in current_edges and (v, u) not in current_edges:
                current_edges.append((u, v))
        
        # Inject anomalies at specific timestamps
        is_anomaly = False
        anomaly_type = "normal"
        
        if t in [15, 30, 45]:  # Anomaly timestamps
            is_anomaly = True
            if t == 15:  # Star burst anomaly
                center_node = np.random.choice(num_nodes)
                star_edges = [(center_node, i) for i in range(num_nodes) if i != center_node]
                current_edges.extend(star_edges[:20])  # Add 20 star edges
                anomaly_type = "star_burst"
                
            elif t == 30:  # Clique anomaly
                clique_nodes = np.random.choice(num_nodes, 10, replace=False)
                clique_edges = [(i, j) for i in clique_nodes for j in clique_nodes if i < j]
                current_edges.extend(clique_edges)
                anomaly_type = "clique"
                
            elif t == 45:  # Bridge removal (disconnection)
                # Remove edges to create disconnection
                edges_to_remove = np.random.choice(len(current_edges), 30, replace=False)
                for idx in sorted(edges_to_remove, reverse=True):
                    current_edges.pop(idx)
                anomaly_type = "disconnection"
        
        # Create node features (degree-based + random)
        G_current = nx.Graph()
        G_current.add_edges_from(current_edges)
        
        node_features = []
        for node in range(num_nodes):
            degree = G_current.degree(node) if node in G_current else 0
            random_feat = np.random.normal(0, 1, 3)  # 3 random features
            features = [degree] + list(random_feat)
            node_features.append(features)
        
        # Store timestamp data
        timestamp_data = {
            'timestamp': t,
            'edges': current_edges,
            'node_features': node_features,
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'num_nodes': num_nodes,
            'num_edges': len(current_edges)
        }
        
        temporal_data.append(timestamp_data)
    
    # Save synthetic data
    import pickle
    with open('data/synthetic/temporal_graph_with_anomalies.pkl', 'wb') as f:
        pickle.dump(temporal_data, f)
    
    print(f"✅ Created synthetic temporal graph:")
    print(f"   - {num_timestamps} timestamps")
    print(f"   - {num_nodes} nodes")
    print(f"   - Anomalies at timestamps: [15, 30, 45]")
    print(f"   - Saved to data/synthetic/temporal_graph_with_anomalies.pkl")
    
    # Create summary CSV
    summary_data = []
    for data in temporal_data:
        summary_data.append({
            'timestamp': data['timestamp'],
            'num_edges': data['num_edges'],
            'is_anomaly': data['is_anomaly'],
            'anomaly_type': data['anomaly_type']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('data/synthetic/temporal_graph_summary.csv', index=False)
    print("✅ Created summary CSV")

def explore_bitcoin_data():
    """Load and explore Bitcoin data structure"""
    print("🔍 Exploring Bitcoin datasets...")
    
    try:
        # Load Bitcoin Alpha
        alpha_path = "data/bitcoin/soc-sign-bitcoin-alpha.csv.gz"
        if os.path.exists(alpha_path):
            df_alpha = pd.read_csv(alpha_path, header=None, 
                                 names=['source', 'target', 'rating', 'timestamp'])
            print(f"\n📊 Bitcoin Alpha stats:")
            print(f"   - Edges: {len(df_alpha)}")
            print(f"   - Nodes: {len(set(df_alpha['source']) | set(df_alpha['target']))}")
            print(f"   - Time range: {df_alpha['timestamp'].min()} to {df_alpha['timestamp'].max()}")
            print(f"   - Ratings: {df_alpha['rating'].value_counts().to_dict()}")
            
            # Save processed version
            df_alpha.to_csv('data/processed/bitcoin_alpha_processed.csv', index=False)
            
        # Load Bitcoin OTC
        otc_path = "data/bitcoin/soc-sign-bitcoin-otc.csv.gz"
        if os.path.exists(otc_path):
            df_otc = pd.read_csv(otc_path, header=None,
                               names=['source', 'target', 'rating', 'timestamp'])
            print(f"\n📊 Bitcoin OTC stats:")
            print(f"   - Edges: {len(df_otc)}")
            print(f"   - Nodes: {len(set(df_otc['source']) | set(df_otc['target']))}")
            print(f"   - Time range: {df_otc['timestamp'].min()} to {df_otc['timestamp'].max()}")
            print(f"   - Ratings: {df_otc['rating'].value_counts().to_dict()}")
            
            # Save processed version
            df_otc.to_csv('data/processed/bitcoin_otc_processed.csv', index=False)
            
    except Exception as e:
        print(f"❌ Error exploring Bitcoin data: {e}")

def create_dataset_info():
    """Create a README file with dataset information"""
    readme_content = """# Temporal GNN Anomaly Detection - Datasets

## Dataset Overview

### 1. Bitcoin Trust Networks
- **Source**: Stanford SNAP
- **Description**: Signed trust networks from Bitcoin trading platforms
- **Files**:
  - `bitcoin/soc-sign-bitcoin-alpha.csv.gz`: Bitcoin Alpha platform
  - `bitcoin/soc-sign-bitcoin-otc.csv.gz`: Bitcoin OTC platform
- **Format**: [source_user, target_user, rating, timestamp]
- **Ratings**: +1 (trust), -1 (distrust)
- **Use case**: Financial fraud detection, trust relationship anomalies

### 2. Synthetic Temporal Graph
- **Source**: Generated
- **Description**: Synthetic temporal graph with injected anomalies
- **Files**:
  - `synthetic/temporal_graph_with_anomalies.pkl`: Full temporal data
  - `synthetic/temporal_graph_summary.csv`: Summary statistics
- **Anomalies**:
  - Timestamp 15: Star burst (one node connects to many)
  - Timestamp 30: Dense clique formation
  - Timestamp 45: Network disconnection
- **Use case**: Algorithm development and testing

### 3. Data Processing
- **Processed files**: `processed/` directory contains cleaned datasets
- **Format**: Standardized CSV files ready for GNN training

## Usage Notes
- All timestamps are Unix epoch time
- Node IDs are integers (may need remapping for GNNs)
- Negative ratings in Bitcoin data can indicate potential fraud
- Synthetic data has ground truth anomaly labels for evaluation

## Next Steps
1. Implement temporal graph loaders
2. Create train/validation/test splits
3. Design anomaly injection strategies for real data
"""
    
    with open('data/README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Created dataset documentation (data/README.md)")

def main():
    """Main function to download and prepare all datasets"""
    print("🚀 Starting dataset preparation for Temporal GNN project...")
    print("=" * 60)
    
    create_directories()
    download_bitcoin_data()
    create_synthetic_temporal_graph()
    explore_bitcoin_data()
    create_dataset_info()
    
    print("=" * 60)
    print("🎉 Dataset preparation complete!")
    print("\nDatasets ready:")
    print("✅ Bitcoin Alpha and OTC networks (real financial data)")
    print("✅ Synthetic temporal graph with known anomalies")
    print("✅ Processed and documented datasets")
    print("\nNext steps:")
    print("1. Run: python test_graph_setup.py (test libraries)")
    print("2. Start implementing static GNN baseline")
    print("3. Review literature (DOMINANT, Temporal Graph Networks)")

if __name__ == "__main__":
    main()
