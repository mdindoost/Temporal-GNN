#!/usr/bin/env python3
"""
Debug script to check the structure of synthetic data
"""

import pandas as pd
import pickle
import os

def check_synthetic_data():
    """Check what's actually in the synthetic data files"""
    
    data_path = 'data/synthetic/'
    
    print("="*60)
    print("CHECKING SYNTHETIC DATA STRUCTURE")
    print("="*60)
    
    # Check if files exist
    pkl_file = os.path.join(data_path, 'temporal_graph_with_anomalies.pkl')
    csv_file = os.path.join(data_path, 'temporal_graph_summary.csv')
    
    print(f"Pickle file exists: {os.path.exists(pkl_file)}")
    print(f"CSV file exists: {os.path.exists(csv_file)}")
    print()
    
    # Check pickle file
    if os.path.exists(pkl_file):
        print("PICKLE FILE CONTENTS:")
        print("-" * 30)
        with open(pkl_file, 'rb') as f:
            temporal_graph = pickle.load(f)
        
        print(f"Type: {type(temporal_graph)}")
        print(f"Length: {len(temporal_graph)}")
        print(f"First few items: {list(temporal_graph.keys())[:10] if hasattr(temporal_graph, 'keys') else 'Not a dict'}")
        
        # Check a few graphs
        if hasattr(temporal_graph, 'keys'):
            for i, (key, graph) in enumerate(list(temporal_graph.items())[:5]):
                print(f"  Time {key}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        elif isinstance(temporal_graph, list):
            for i, item in enumerate(temporal_graph[:5]):
                if hasattr(item, 'number_of_nodes'):
                    # It's a NetworkX graph
                    print(f"  Time {i}: {item.number_of_nodes()} nodes, {item.number_of_edges()} edges")
                else:
                    # It's something else - let's see what
                    print(f"  Time {i}: Type = {type(item)}")
                    if isinstance(item, dict):
                        print(f"    Dict keys: {list(item.keys())[:10]}")
                    elif hasattr(item, '__len__'):
                        print(f"    Length: {len(item)}")
                    else:
                        print(f"    Value: {str(item)[:100]}")
        print()
    
    # Check CSV file
    if os.path.exists(csv_file):
        print("CSV FILE CONTENTS:")
        print("-" * 30)
        df = pd.read_csv(csv_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print()
        
        # Check for anomaly information
        possible_anomaly_cols = ['has_anomaly', 'anomaly', 'is_anomaly', 'anomalous']
        for col in possible_anomaly_cols:
            if col in df.columns:
                print(f"Found anomaly column: {col}")
                print(f"Anomaly timestamps: {df[df[col]]['timestamp'].tolist() if 'timestamp' in df.columns else 'No timestamp column'}")
    
    # List all files in the directory
    print("ALL FILES IN SYNTHETIC DATA DIRECTORY:")
    print("-" * 40)
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} bytes")
    else:
        print("Directory does not exist!")

if __name__ == "__main__":
    check_synthetic_data()
