#!/usr/bin/env python3
"""
Bitcoin Trust Network Data Downloader
Downloads and prepares Bitcoin Alpha and OTC trust networks from Stanford SNAP
"""

import os
import requests
import gzip
import pandas as pd
from tqdm import tqdm
import numpy as np

def download_file(url, filepath, description="Downloading"):
    """Download file with progress bar"""
    print(f"📥 {description}...")
    print(f"   URL: {url}")
    print(f"   Target: {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✅ Downloaded: {filepath}")
    return filepath

def download_bitcoin_networks():
    """Download Bitcoin Alpha and OTC trust networks"""
    print("🏗️ BITCOIN TRUST NETWORK DOWNLOADER")
    print("="*60)
    
    # Create data directory
    base_dir = "data/bitcoin"
    os.makedirs(base_dir, exist_ok=True)
    
    # Bitcoin network URLs from Stanford SNAP (updated URLs)
    networks = {
        'alpha': {
            'url': 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
            'filename': 'soc-sign-bitcoin-alpha.csv.gz',
            'description': 'Bitcoin Alpha trust network'
        },
        'otc': {
            'url': 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            'filename': 'soc-sign-bitcoin-otc.csv.gz', 
            'description': 'Bitcoin OTC trust network'
        }
    }
    
    downloaded_files = {}
    
    for network_name, info in networks.items():
        filepath = os.path.join(base_dir, info['filename'])
        
        if os.path.exists(filepath):
            print(f"✅ {info['description']} already exists: {filepath}")
        else:
            try:
                download_file(info['url'], filepath, info['description'])
                downloaded_files[network_name] = filepath
            except Exception as e:
                print(f"❌ Failed to download {network_name}: {e}")
                continue
    
    return downloaded_files

def analyze_bitcoin_data(filepath, network_name):
    """Analyze downloaded Bitcoin data"""
    print(f"\n🔍 Analyzing {network_name} network...")
    
    try:
        # Load the data
        df = pd.read_csv(filepath, compression='gzip', header=None,
                        names=['source', 'target', 'rating', 'timestamp'])
        
        print(f"📊 {network_name.upper()} Network Statistics:")
        print(f"   Total edges: {len(df):,}")
        print(f"   Unique users: {len(set(df['source'].tolist() + df['target'].tolist())):,}")
        print(f"   Time range: {pd.to_datetime(df['timestamp'], unit='s').min()} to {pd.to_datetime(df['timestamp'], unit='s').max()}")
        
        # Rating analysis
        rating_counts = df['rating'].value_counts().sort_index()
        print(f"   Rating distribution:")
        for rating, count in rating_counts.items():
            percentage = (count / len(df)) * 100
            print(f"     {rating:+2d}: {count:6,} ({percentage:5.1f}%)")
        
        # Negative ratings (potential fraud indicators)
        negative_edges = df[df['rating'] < 0]
        print(f"   Negative ratings: {len(negative_edges):,} ({len(negative_edges)/len(df)*100:.1f}%)")
        
        # Temporal analysis
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        monthly_activity = df.groupby(df['datetime'].dt.to_period('M')).size()
        print(f"   Active months: {len(monthly_activity)}")
        print(f"   Peak activity: {monthly_activity.max():,} edges in {monthly_activity.idxmax()}")
        
        # User activity analysis
        user_activity = {}
        for user in set(df['source'].tolist() + df['target'].tolist()):
            user_edges = df[(df['source'] == user) | (df['target'] == user)]
            negative_received = len(df[(df['target'] == user) & (df['rating'] < 0)])
            total_received = len(df[df['target'] == user])
            
            if total_received > 0:
                negative_ratio = negative_received / total_received
                if negative_ratio > 0.5:  # More than 50% negative
                    user_activity[user] = {
                        'total_interactions': len(user_edges),
                        'negative_received': negative_received,
                        'negative_ratio': negative_ratio
                    }
        
        print(f"   Suspicious users (>50% negative ratings): {len(user_activity)}")
        
        return {
            'dataframe': df,
            'total_edges': len(df),
            'unique_users': len(set(df['source'].tolist() + df['target'].tolist())),
            'negative_edges': len(negative_edges),
            'suspicious_users': len(user_activity),
            'time_range': (df['timestamp'].min(), df['timestamp'].max()),
            'monthly_activity': monthly_activity
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {network_name}: {e}")
        return None

def create_processed_datasets():
    """Create processed versions of Bitcoin data for easier loading"""
    print(f"\n🔧 Creating processed datasets...")
    
    base_dir = "data/bitcoin"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    networks = ['alpha', 'otc']
    processed_files = {}
    
    for network in networks:
        raw_file = os.path.join(base_dir, f'soc-sign-bitcoin-{network}.csv.gz')
        
        if not os.path.exists(raw_file):
            print(f"⚠️ Raw file not found: {raw_file}")
            continue
            
        try:
            # Load raw data
            df = pd.read_csv(raw_file, compression='gzip', header=None,
                           names=['source', 'target', 'rating', 'timestamp'])
            
            # Basic preprocessing
            df = df.dropna()
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create node mapping
            all_nodes = list(set(df['source'].tolist() + df['target'].tolist()))
            node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
            
            df['source_idx'] = df['source'].map(node_mapping)
            df['target_idx'] = df['target'].map(node_mapping)
            
            # Add temporal features
            df = df.sort_values('timestamp')
            df['month'] = df['datetime'].dt.to_period('M')
            
            # Save processed data
            processed_file = os.path.join(processed_dir, f'bitcoin_{network}_processed.csv')
            df.to_csv(processed_file, index=False)
            processed_files[network] = processed_file
            
            # Save node mapping
            mapping_file = os.path.join(processed_dir, f'bitcoin_{network}_node_mapping.csv')
            pd.DataFrame(list(node_mapping.items()), 
                        columns=['original_id', 'node_idx']).to_csv(mapping_file, index=False)
            
            print(f"✅ Processed {network}: {processed_file}")
            print(f"   Nodes: {len(all_nodes):,}, Edges: {len(df):,}")
            
        except Exception as e:
            print(f"❌ Error processing {network}: {e}")
    
    return processed_files

def validate_download():
    """Validate that downloads completed successfully"""
    print(f"\n✅ VALIDATION RESULTS:")
    print("="*40)
    
    files_to_check = [
        "data/bitcoin/soc-sign-bitcoin-alpha.csv.gz",
        "data/bitcoin/soc-sign-bitcoin-otc.csv.gz",
        "data/processed/bitcoin_alpha_processed.csv",
        "data/processed/bitcoin_otc_processed.csv"
    ]
    
    all_good = True
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ {filepath} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {filepath} - MISSING")
            all_good = False
    
    if all_good:
        print(f"\n🎉 ALL FILES READY FOR COMPREHENSIVE TESTING!")
        print(f"📋 Next steps:")
        print(f"   1. Update comprehensive_testing.py bitcoin paths")
        print(f"   2. Run: python comprehensive_testing.py")
    else:
        print(f"\n⚠️ Some files missing - check errors above")
    
    return all_good

def main():
    """Main download and setup pipeline"""
    print("🏗️ BITCOIN TRUST NETWORK SETUP")
    print("="*50)
    
    # Step 1: Download raw data
    downloaded = download_bitcoin_networks()
    
    if not downloaded:
        print("❌ No files downloaded - check network connection")
        return
    
    # Step 2: Analyze downloaded data
    for network_name, filepath in downloaded.items():
        if os.path.exists(filepath):
            analyze_bitcoin_data(filepath, network_name)
    
    # Step 3: Create processed versions
    processed = create_processed_datasets()
    
    # Step 4: Validate everything
    validate_download()
    
    print(f"\n🎯 SETUP COMPLETE!")
    print(f"Ready for comprehensive testing with real Bitcoin fraud detection data.")

if __name__ == "__main__":
    main()
