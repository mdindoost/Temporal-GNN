#!/usr/bin/env python3
"""
CORRECTED PHASE 1: Data Foundation - Multiple Dataset Scenarios
Author: Paper Verification Team
Purpose: Test different dataset combinations to match paper claims

CRITICAL FINDINGS:
- Paper claims 24,186 edges but combined data has 59,778 edges
- This suggests paper used ONLY Bitcoin Alpha (24,186 edges)
- Need to test both scenarios to identify correct methodology
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CorrectedDataVerifier:
    """
    Test multiple dataset scenarios to match paper claims
    """
    
    def __init__(self, data_dir="/home/md724/temporal-gnn-project/data/bitcoin"):
        self.data_dir = data_dir
        self.alpha_file = f"{data_dir}/soc-sign-bitcoin-alpha.csv.gz"
        self.otc_file = f"{data_dir}/soc-sign-bitcoin-otc.csv.gz"
        
        # Paper claims to verify
        self.paper_claims = {
            'total_edges': 24186,
            'total_users': 3783,
            'suspicious_users': 73,
            'negative_threshold': 0.30,
            'min_interactions': 5
        }
        
    def load_networks_separately(self):
        """Load networks separately to test different scenarios"""
        print("🔍 CORRECTED PHASE 1.1: Loading Networks Separately...")
        
        # Load Bitcoin Alpha
        alpha_df = pd.read_csv(self.alpha_file, names=['source', 'target', 'rating', 'timestamp'])
        print(f"✅ Bitcoin Alpha: {len(alpha_df)} edges")
        
        # Load Bitcoin OTC
        otc_df = pd.read_csv(self.otc_file, names=['source', 'target', 'rating', 'timestamp'])
        print(f"✅ Bitcoin OTC: {len(otc_df)} edges")
        
        # Combined
        combined_df = pd.concat([alpha_df, otc_df], ignore_index=True)
        print(f"✅ Combined: {len(combined_df)} edges")
        
        return {
            'alpha_only': alpha_df,
            'otc_only': otc_df,
            'combined': combined_df
        }
        
    def test_scenario(self, name, df):
        """Test a specific dataset scenario against paper claims"""
        print(f"\n🧪 TESTING SCENARIO: {name.upper()}")
        print("="*50)
        
        # Basic statistics
        total_edges = len(df)
        unique_users = len(set(df['source'].unique()) | set(df['target'].unique()))
        
        print(f"📊 Basic Statistics:")
        print(f"   Edges: {total_edges}")
        print(f"   Users: {unique_users}")
        
        # Check against paper claims
        edges_match = total_edges == self.paper_claims['total_edges']
        users_match = unique_users == self.paper_claims['total_users']
        
        print(f"✅ Edges match paper: {edges_match}")
        print(f"✅ Users match paper: {users_match}")
        
        # Find suspicious users
        suspicious_count, suspicious_analysis = self.find_suspicious_users(df)
        suspicious_match = suspicious_count == self.paper_claims['suspicious_users']
        
        print(f"✅ Suspicious users match paper: {suspicious_match}")
        
        scenario_results = {
            'dataset': name,
            'total_edges': total_edges,
            'total_users': unique_users,
            'suspicious_users': suspicious_count,
            'edges_match': edges_match,
            'users_match': users_match,
            'suspicious_match': suspicious_match,
            'overall_match': edges_match and users_match and suspicious_match,
            'suspicious_analysis': suspicious_analysis[:10]  # Top 10 for brevity
        }
        
        return scenario_results
        
    def find_suspicious_users(self, df):
        """Find suspicious users in a dataset"""
        # Calculate user statistics (as targets - receiving ratings)
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0, 'positive': 0})
        
        for _, row in df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            if rating == -1:
                user_stats[target]['negative'] += 1
            elif rating == 1:
                user_stats[target]['positive'] += 1
                
        # Apply paper criteria
        suspicious_users = []
        user_analysis = []
        
        for user_id, stats in user_stats.items():
            if stats['total'] >= self.paper_claims['min_interactions']:
                negative_ratio = stats['negative'] / stats['total']
                is_suspicious = negative_ratio > self.paper_claims['negative_threshold']
                
                user_analysis.append({
                    'user_id': user_id,
                    'total_interactions': stats['total'],
                    'negative_ratings': stats['negative'],
                    'positive_ratings': stats['positive'],
                    'negative_ratio': negative_ratio,
                    'is_suspicious': is_suspicious
                })
                
                if is_suspicious:
                    suspicious_users.append(user_id)
                    
        # Sort by negative ratio
        user_analysis.sort(key=lambda x: x['negative_ratio'], reverse=True)
        
        print(f"📊 Suspicious User Analysis:")
        print(f"   Users with ≥{self.paper_claims['min_interactions']} interactions: {len(user_analysis)}")
        print(f"   Suspicious users (>{self.paper_claims['negative_threshold']*100}% negative): {len(suspicious_users)}")
        
        # Show top suspicious users
        print(f"🔴 Top 5 Suspicious Users:")
        for i, user in enumerate(user_analysis[:5]):
            if user['is_suspicious']:
                print(f"   {i+1}. User {user['user_id']}: {user['negative_ratio']:.3f} "
                      f"({user['negative_ratings']}/{user['total_interactions']})")
                      
        return len(suspicious_users), user_analysis
        
    def run_comprehensive_test(self):
        """Test all scenarios to find the correct methodology"""
        print("🚀 CORRECTED PHASE 1: COMPREHENSIVE DATASET TESTING")
        print("="*60)
        
        # Load all networks
        networks = self.load_networks_separately()
        
        # Test each scenario
        results = {}
        
        print(f"\n🎯 TESTING AGAINST PAPER CLAIMS:")
        print(f"   Expected Edges: {self.paper_claims['total_edges']}")
        print(f"   Expected Users: {self.paper_claims['total_users']}")
        print(f"   Expected Suspicious: {self.paper_claims['suspicious_users']}")
        
        for scenario_name, df in networks.items():
            results[scenario_name] = self.test_scenario(scenario_name, df)
            
        # Summary analysis
        print(f"\n🎯 SCENARIO COMPARISON SUMMARY:")
        print("="*60)
        
        perfect_matches = []
        partial_matches = []
        
        for scenario, result in results.items():
            status = "🎯 PERFECT MATCH" if result['overall_match'] else "⚠️  PARTIAL MATCH"
            print(f"{status}: {scenario.upper()}")
            print(f"   Edges: {result['total_edges']} ({'✅' if result['edges_match'] else '❌'})")
            print(f"   Users: {result['total_users']} ({'✅' if result['users_match'] else '❌'})")
            print(f"   Suspicious: {result['suspicious_users']} ({'✅' if result['suspicious_match'] else '❌'})")
            
            if result['overall_match']:
                perfect_matches.append(scenario)
            elif any([result['edges_match'], result['users_match'], result['suspicious_match']]):
                partial_matches.append(scenario)
                
        # Conclusion
        print(f"\n🔍 CONCLUSION:")
        if perfect_matches:
            print(f"✅ PERFECT MATCH FOUND: {', '.join(perfect_matches).upper()}")
            print(f"   Paper methodology uses: {perfect_matches[0].replace('_', ' ').title()}")
        elif partial_matches:
            print(f"⚠️  PARTIAL MATCHES FOUND: {', '.join(partial_matches).upper()}")
            print(f"   Investigate data preprocessing differences")
        else:
            print(f"❌ NO MATCHES FOUND - SIGNIFICANT METHODOLOGY DIFFERENCES")
            
        # Save corrected results
        self.save_corrected_results(results)
        
        return results, perfect_matches
        
    def save_corrected_results(self, results):
        """Save corrected verification results"""
        print(f"\n💾 Saving Corrected Verification Results...")
        
        # Convert to serializable format
        serializable_results = {}
        for scenario, result in results.items():
            serializable_results[scenario] = {
                'dataset': result['dataset'],
                'total_edges': int(result['total_edges']),
                'total_users': int(result['total_users']),
                'suspicious_users': int(result['suspicious_users']),
                'edges_match': bool(result['edges_match']),
                'users_match': bool(result['users_match']),
                'suspicious_match': bool(result['suspicious_match']),
                'overall_match': bool(result['overall_match']),
                'top_suspicious': result['suspicious_analysis'][:5]  # Top 5 only
            }
            
        corrected_report = {
            'timestamp': datetime.now().isoformat(),
            'paper_claims': self.paper_claims,
            'scenario_results': serializable_results,
            'discrepancies_identified': True,
            'recommended_action': 'Use scenario that matches paper claims or investigate preprocessing'
        }
        
        # Save to file
        with open('data_verification/corrected_verification_report.json', 'w') as f:
            json.dump(corrected_report, f, indent=2)
            
        print(f"✅ Saved: corrected_verification_report.json")

def main():
    """Main corrected verification execution"""
    verifier = CorrectedDataVerifier()
    results, perfect_matches = verifier.run_comprehensive_test()
    
    print(f"\n🎯 NEXT STEPS:")
    if perfect_matches:
        print(f"1. Use {perfect_matches[0]} dataset for all subsequent verification")
        print(f"2. Update paper claims or methodology description")
        print(f"3. Proceed to PHASE 2 with correct dataset")
    else:
        print(f"1. Investigate data preprocessing differences")
        print(f"2. Check if paper used different filtering criteria")
        print(f"3. Verify timestamp ranges or data cleaning steps")
        
if __name__ == "__main__":
    main()
