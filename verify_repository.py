#!/usr/bin/env python3
"""
Quick Repository Verification Script
Run this to verify your repository setup
"""

import os
import sys

def verify_repository():
    """Verify repository structure and key files"""
    
    print("🔍 REPOSITORY VERIFICATION")
    print("=" * 30)
    
    # Check key files
    key_files = [
        "README.md",
        "LICENSE", 
        "requirements.txt",
        "temporal_anomaly_detector.py",
        "bitcoin_baseline_comparison.py",
        "set_temporal_gnn"
    ]
    
    print("\n📁 Key Files:")
    missing_files = []
    
    for file in key_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} (missing)")
            missing_files.append(file)
    
    # Check directories
    key_dirs = [
        "data/processed",
        "multi_dataset_expansion/results",
        "logs",
        "src"
    ]
    
    print("\n📂 Key Directories:")
    missing_dirs = []
    
    for dir_path in key_dirs:
        if os.path.exists(dir_path):
            files_count = len(os.listdir(dir_path)) if os.path.isdir(dir_path) else 0
            print(f"   ✅ {dir_path}/ ({files_count} items)")
        else:
            print(f"   ❌ {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    # Check data integrity
    print("\n🔬 Data Integrity:")
    
    try:
        import pandas as pd
        
        datasets = [
            ("data/processed/bitcoin_alpha_processed.csv", 73),
            ("data/processed/bitcoin_otc_processed.csv", 219)
        ]
        
        for dataset_path, expected_suspicious in datasets:
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                print(f"   ✅ {dataset_path}: {len(df)} edges")
                
                # Quick ground truth check
                from collections import defaultdict
                user_stats = defaultdict(lambda: {'total': 0, 'negative': 0})
                for _, row in df.iterrows():
                    target = row['target_idx']
                    user_stats[target]['total'] += 1
                    if row['rating'] < 0:
                        user_stats[target]['negative'] += 1
                
                suspicious_users = set()
                for user, stats in user_stats.items():
                    if stats['total'] >= 5 and stats['negative'] / stats['total'] > 0.3:
                        suspicious_users.add(user)
                
                actual_count = len(suspicious_users)
                if actual_count == expected_suspicious:
                    print(f"      ✅ Ground truth: {actual_count} suspicious users")
                else:
                    print(f"      ⚠️  Ground truth: {actual_count} (expected {expected_suspicious})")
            else:
                print(f"   ❌ {dataset_path}: Not found")
                
    except ImportError:
        print("   ⚠️  pandas not available - skipping data checks")
    except Exception as e:
        print(f"   ❌ Data check error: {e}")
    
    # Check results
    print("\n📊 Results:")
    
    results_file = "multi_dataset_expansion/results/final_expansion_results.json"
    if os.path.exists(results_file):
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            datasets_count = len(results)
            methods_count = len(results[list(results.keys())[0]]) if results else 0
            
            print(f"   ✅ Results available: {datasets_count} datasets, {methods_count} methods")
            
            # Check key findings
            if 'bitcoin_alpha' in results and 'bitcoin_otc' in results:
                alpha_sep = results['bitcoin_alpha'].get('negative_ratio', {}).get('separation_ratio', 0)
                otc_sep = results['bitcoin_otc'].get('negative_ratio', {}).get('separation_ratio', 0)
                
                if alpha_sep > 20 and otc_sep > 20:
                    consistency = abs(alpha_sep - otc_sep) / max(alpha_sep, otc_sep) * 100
                    print(f"   ✅ Cross-dataset consistency: {consistency:.1f}% difference")
                else:
                    print(f"   ⚠️  Separation ratios seem low: {alpha_sep:.1f}×, {otc_sep:.1f}×")
            
        except Exception as e:
            print(f"   ❌ Results check error: {e}")
    else:
        print(f"   ⚠️  Results not found - run expansion first")
    
    # Summary
    print("\n🎯 VERIFICATION SUMMARY:")
    
    if not missing_files and not missing_dirs:
        print("   ✅ Repository structure complete")
    else:
        print("   ⚠️  Some files/directories missing")
        if missing_files:
            print(f"      Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"      Missing dirs: {', '.join(missing_dirs)}")
    
    print("\n🚀 Next steps:")
    print("   1. Run: source set_temporal_gnn")
    print("   2. Run: cd multi_dataset_expansion && sbatch run_tgb_expansion.slurm")
    print("   3. Check: multi_dataset_expansion/results/")

if __name__ == "__main__":
    verify_repository()
