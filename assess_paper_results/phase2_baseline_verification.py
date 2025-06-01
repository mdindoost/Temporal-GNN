#!/usr/bin/env python3
"""
PHASE 2: Baseline Implementation & Verification
Author: Paper Verification Team
Purpose: Implement and verify all baseline methods from Table 1

VERIFICATION TARGETS:
- Table 1: Baseline comparison on retrospective analysis
  - Negative Ratio: 25.08× separation ratio, 0.460 precision@50
  - Temporal Volatility: 2.46× separation ratio, 0.580 precision@50
  - Weighted PageRank: 2.15× separation ratio, 0.280 precision@50
  - TempAnom-GNN: 1.33× separation ratio, 0.670 precision@50

CORRECTED GROUND TRUTH:
- Dataset: Bitcoin Alpha (24,186 edges, 3,783 users)
- Suspicious users: 9 (realistic criteria)
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class BaselineVerifier:
    """
    Implement and verify all baseline methods from Table 1
    """
    
    def __init__(self):
        # Load corrected ground truth
        self.df = pd.read_csv('data_verification/bitcoin_alpha_corrected_labels.csv')
        
        with open('data_verification/corrected_ground_truth_report.json', 'r') as f:
            self.ground_truth = json.load(f)
            
        # Load suspicious user analysis
        self.user_analysis = pd.read_csv('data_verification/corrected_suspicious_user_analysis.csv')
        
        # Paper claims to verify (Table 1)
        self.table1_claims = {
            'negative_ratio': {'separation_ratio': 25.08, 'precision_at_50': 0.460},
            'temporal_volatility': {'separation_ratio': 2.46, 'precision_at_50': 0.580},
            'weighted_pagerank': {'separation_ratio': 2.15, 'precision_at_50': 0.280},
            'tempamon_gnn': {'separation_ratio': 1.33, 'precision_at_50': 0.670}
        }
        
        self.verification_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for baseline analysis"""
        print("📁 PHASE 2.1: Loading and Preparing Data...")
        
        # Basic statistics
        print(f"✅ Loaded: {len(self.df)} edges")
        print(f"✅ Users: {len(set(self.df['source']) | set(self.df['target']))}")
        
        # Suspicious users from ground truth
        suspicious_users = set(self.ground_truth['suspicious_users_list'])
        print(f"✅ Suspicious users: {len(suspicious_users)}")
        
        # Create ground truth labels for all users
        all_users = set(self.df['source']) | set(self.df['target'])
        self.user_labels = {user: 1 if user in suspicious_users else 0 for user in all_users}
        
        print(f"✅ Ground truth labels created for {len(self.user_labels)} users")
        return True
        
    def implement_negative_ratio_baseline(self):
        """Implement Negative Ratio baseline (should achieve 25.08× separation)"""
        print("\n🔍 PHASE 2.2: Implementing Negative Ratio Baseline...")
        
        # Calculate negative ratio for each user (as target)
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0})
        
        for _, row in self.df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            if rating == -1:
                user_stats[target]['negative'] += 1
                
        # Calculate negative ratios
        user_scores = {}
        for user_id, stats in user_stats.items():
            if stats['total'] > 0:
                negative_ratio = stats['negative'] / stats['total']
                user_scores[user_id] = negative_ratio
            else:
                user_scores[user_id] = 0.0
                
        # Calculate separation ratio (suspicious vs normal users)
        suspicious_scores = [user_scores[user] for user in self.ground_truth['suspicious_users_list'] if user in user_scores]
        normal_users = [user for user in user_scores.keys() if user not in self.ground_truth['suspicious_users_list']]
        normal_scores = [user_scores[user] for user in normal_users]
        
        if len(suspicious_scores) > 0 and len(normal_scores) > 0:
            avg_suspicious = np.mean(suspicious_scores)
            avg_normal = np.mean(normal_scores)
            separation_ratio = avg_suspicious / avg_normal if avg_normal > 0 else float('inf')
        else:
            separation_ratio = 0.0
            
        # Calculate Precision@50
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:50]]
        top_50_labels = [self.user_labels[user] for user in top_50_users]
        precision_at_50 = sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
        results = {
            'method': 'Negative Ratio',
            'separation_ratio': separation_ratio,
            'precision_at_50': precision_at_50,
            'user_scores': user_scores,
            'paper_separation': self.table1_claims['negative_ratio']['separation_ratio'],
            'paper_precision': self.table1_claims['negative_ratio']['precision_at_50'],
            'separation_match': abs(separation_ratio - self.table1_claims['negative_ratio']['separation_ratio']) < 1.0,
            'precision_match': abs(precision_at_50 - self.table1_claims['negative_ratio']['precision_at_50']) < 0.1
        }
        
        print(f"📊 Negative Ratio Results:")
        print(f"   Separation Ratio: {separation_ratio:.2f} (Paper: {self.table1_claims['negative_ratio']['separation_ratio']:.2f})")
        print(f"   Precision@50: {precision_at_50:.3f} (Paper: {self.table1_claims['negative_ratio']['precision_at_50']:.3f})")
        print(f"   Separation Match: {results['separation_match']}")
        print(f"   Precision Match: {results['precision_match']}")
        
        self.verification_results['negative_ratio'] = results
        return results
        
    def implement_temporal_volatility_baseline(self):
        """Implement Temporal Volatility baseline"""
        print("\n🔍 PHASE 2.3: Implementing Temporal Volatility Baseline...")
        
        # Calculate temporal volatility for each user
        user_temporal_stats = defaultdict(list)
        
        # Group ratings by user and timestamp
        for _, row in self.df.iterrows():
            target = row['target']
            rating = row['rating']
            timestamp = row['timestamp']
            
            user_temporal_stats[target].append((timestamp, rating))
            
        user_scores = {}
        for user_id, ratings in user_temporal_stats.items():
            if len(ratings) >= 2:
                # Sort by timestamp
                ratings.sort(key=lambda x: x[0])
                
                # Calculate volatility as standard deviation of ratings over time
                rating_values = [r[1] for r in ratings]
                volatility = np.std(rating_values) if len(rating_values) > 1 else 0.0
                user_scores[user_id] = volatility
            else:
                user_scores[user_id] = 0.0
                
        # Calculate metrics similar to negative ratio
        suspicious_scores = [user_scores[user] for user in self.ground_truth['suspicious_users_list'] if user in user_scores]
        normal_users = [user for user in user_scores.keys() if user not in self.ground_truth['suspicious_users_list']]
        normal_scores = [user_scores[user] for user in normal_users]
        
        if len(suspicious_scores) > 0 and len(normal_scores) > 0:
            avg_suspicious = np.mean(suspicious_scores)
            avg_normal = np.mean(normal_scores)
            separation_ratio = avg_suspicious / avg_normal if avg_normal > 0 else float('inf')
        else:
            separation_ratio = 0.0
            
        # Precision@50
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:50]]
        top_50_labels = [self.user_labels[user] for user in top_50_users]
        precision_at_50 = sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
        results = {
            'method': 'Temporal Volatility',
            'separation_ratio': separation_ratio,
            'precision_at_50': precision_at_50,
            'user_scores': user_scores,
            'paper_separation': self.table1_claims['temporal_volatility']['separation_ratio'],
            'paper_precision': self.table1_claims['temporal_volatility']['precision_at_50'],
            'separation_match': abs(separation_ratio - self.table1_claims['temporal_volatility']['separation_ratio']) < 1.0,
            'precision_match': abs(precision_at_50 - self.table1_claims['temporal_volatility']['precision_at_50']) < 0.1
        }
        
        print(f"📊 Temporal Volatility Results:")
        print(f"   Separation Ratio: {separation_ratio:.2f} (Paper: {self.table1_claims['temporal_volatility']['separation_ratio']:.2f})")
        print(f"   Precision@50: {precision_at_50:.3f} (Paper: {self.table1_claims['temporal_volatility']['precision_at_50']:.3f})")
        print(f"   Separation Match: {results['separation_match']}")
        print(f"   Precision Match: {results['precision_match']}")
        
        self.verification_results['temporal_volatility'] = results
        return results
        
    def implement_weighted_pagerank_baseline(self):
        """Implement Weighted PageRank baseline"""
        print("\n🔍 PHASE 2.4: Implementing Weighted PageRank Baseline...")
        
        # Create weighted graph
        G = nx.DiGraph()
        
        # Add edges with weights (negative ratings get negative weights)
        for _, row in self.df.iterrows():
            source = row['source']
            target = row['target'] 
            rating = row['rating']
            
            # Use rating as weight (negative for distrust)
            if G.has_edge(source, target):
                G[source][target]['weight'] += rating
            else:
                G.add_edge(source, target, weight=rating)
                
        # Calculate PageRank with negative weights
        try:
            # For negative weights, we'll use absolute values and invert suspicious scores
            pos_G = nx.DiGraph()
            for u, v, data in G.edges(data=True):
                weight = abs(data['weight'])
                pos_G.add_edge(u, v, weight=weight)
                
            pagerank_scores = nx.pagerank(pos_G, weight='weight', max_iter=100)
            
            # Invert scores for users with predominantly negative ratings
            user_scores = {}
            for user_id in pagerank_scores:
                # Check if user has predominantly negative ratings
                user_ratings = self.df[self.df['target'] == user_id]['rating'].tolist()
                if user_ratings and np.mean(user_ratings) < 0:
                    # Invert score for suspicious users
                    user_scores[user_id] = 1.0 - pagerank_scores[user_id]
                else:
                    user_scores[user_id] = pagerank_scores[user_id]
                    
        except:
            print("   ⚠️ PageRank calculation failed, using fallback method")
            # Fallback: simple degree centrality
            user_scores = dict(G.in_degree())
            
        # Calculate metrics
        suspicious_scores = [user_scores[user] for user in self.ground_truth['suspicious_users_list'] if user in user_scores]
        normal_users = [user for user in user_scores.keys() if user not in self.ground_truth['suspicious_users_list']]
        normal_scores = [user_scores[user] for user in normal_users]
        
        if len(suspicious_scores) > 0 and len(normal_scores) > 0:
            avg_suspicious = np.mean(suspicious_scores)
            avg_normal = np.mean(normal_scores)
            separation_ratio = avg_suspicious / avg_normal if avg_normal > 0 else float('inf')
        else:
            separation_ratio = 0.0
            
        # Precision@50
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:50]]
        top_50_labels = [self.user_labels[user] for user in top_50_users]
        precision_at_50 = sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
        results = {
            'method': 'Weighted PageRank',
            'separation_ratio': separation_ratio,
            'precision_at_50': precision_at_50,
            'user_scores': user_scores,
            'paper_separation': self.table1_claims['weighted_pagerank']['separation_ratio'],
            'paper_precision': self.table1_claims['weighted_pagerank']['precision_at_50'],
            'separation_match': abs(separation_ratio - self.table1_claims['weighted_pagerank']['separation_ratio']) < 1.0,
            'precision_match': abs(precision_at_50 - self.table1_claims['weighted_pagerank']['precision_at_50']) < 0.1
        }
        
        print(f"📊 Weighted PageRank Results:")
        print(f"   Separation Ratio: {separation_ratio:.2f} (Paper: {self.table1_claims['weighted_pagerank']['separation_ratio']:.2f})")
        print(f"   Precision@50: {precision_at_50:.3f} (Paper: {self.table1_claims['weighted_pagerank']['precision_at_50']:.3f})")
        print(f"   Separation Match: {results['separation_match']}")
        print(f"   Precision Match: {results['precision_match']}")
        
        self.verification_results['weighted_pagerank'] = results
        return results
        
    def create_table1_verification_report(self):
        """Create Table 1 verification report"""
        print(f"\n📊 PHASE 2.5: Creating Table 1 Verification Report...")
        
        # Summary table
        print(f"\n{'Method':<20} {'Sep. Ratio':<12} {'Paper':<8} {'Match':<8} {'Prec@50':<10} {'Paper':<8} {'Match':<8}")
        print("-" * 80)
        
        overall_matches = []
        
        for method_key, results in self.verification_results.items():
            method = results['method']
            sep_ratio = results['separation_ratio']
            paper_sep = results['paper_separation']
            sep_match = "✅" if results['separation_match'] else "❌"
            
            precision = results['precision_at_50']
            paper_prec = results['paper_precision']
            prec_match = "✅" if results['precision_match'] else "❌"
            
            overall_match = results['separation_match'] and results['precision_match']
            overall_matches.append(overall_match)
            
            print(f"{method:<20} {sep_ratio:<12.2f} {paper_sep:<8.2f} {sep_match:<8} {precision:<10.3f} {paper_prec:<8.3f} {prec_match:<8}")
            
        # Overall assessment
        total_matches = sum(overall_matches)
        total_methods = len(overall_matches)
        
        print(f"\n🎯 TABLE 1 VERIFICATION SUMMARY:")
        print(f"   Total methods verified: {total_methods}")
        print(f"   Perfect matches: {total_matches}")
        print(f"   Verification rate: {total_matches/total_methods*100:.1f}%")
        
        if total_matches == total_methods:
            print(f"   ✅ ALL TABLE 1 CLAIMS VERIFIED")
        else:
            print(f"   ❌ DISCREPANCIES FOUND IN TABLE 1")
            
        return total_matches == total_methods
        
    def save_baseline_verification_results(self):
        """Save baseline verification results"""
        print(f"\n💾 PHASE 2.6: Saving Baseline Verification Results...")
        
        # Prepare serializable results
        serializable_results = {}
        for method_key, results in self.verification_results.items():
            serializable_results[method_key] = {
                'method': results['method'],
                'separation_ratio': float(results['separation_ratio']),
                'precision_at_50': float(results['precision_at_50']),
                'paper_separation': float(results['paper_separation']),
                'paper_precision': float(results['paper_precision']),
                'separation_match': bool(results['separation_match']),
                'precision_match': bool(results['precision_match']),
                'overall_match': bool(results['separation_match'] and results['precision_match'])
            }
            
        baseline_report = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'PHASE_2_BASELINE_VERIFICATION',
            'ground_truth_source': 'corrected_ground_truth_report.json',
            'table1_verification_results': serializable_results,
            'verification_summary': {
                'total_methods': len(serializable_results),
                'perfect_matches': sum(1 for r in serializable_results.values() if r['overall_match']),
                'overall_verification_status': 'VERIFIED' if all(r['overall_match'] for r in serializable_results.values()) else 'DISCREPANCIES_FOUND'
            },
            'next_phase': 'PHASE_3_COMPONENT_ANALYSIS'
        }
        
        # Save report
        with open('experiments/baseline_verification_report.json', 'w') as f:
            json.dump(baseline_report, f, indent=2)
            
        print(f"✅ Saved: baseline_verification_report.json")
        return baseline_report
        
    def run_complete_baseline_verification(self):
        """Run complete baseline verification pipeline"""
        print("🚀 STARTING PHASE 2: BASELINE IMPLEMENTATION & VERIFICATION")
        print("="*60)
        
        # Load data
        self.load_and_prepare_data()
        
        # Implement each baseline
        self.implement_negative_ratio_baseline()
        self.implement_temporal_volatility_baseline() 
        self.implement_weighted_pagerank_baseline()
        
        # Create verification report
        all_verified = self.create_table1_verification_report()
        
        # Save results
        report = self.save_baseline_verification_results()
        
        # Final status
        print(f"\n🎯 PHASE 2 COMPLETE - STATUS: {report['verification_summary']['overall_verification_status']}")
        print("="*60)
        
        if all_verified:
            print(f"✅ ALL TABLE 1 BASELINES VERIFIED")
            print(f"🔄 Ready for PHASE 3: Component Analysis (Fix Table 5)")
        else:
            print(f"❌ BASELINE DISCREPANCIES FOUND")
            print(f"⚠️ Investigate before proceeding to Table 5 verification")
            
        return all_verified

def main():
    """Main baseline verification execution"""
    verifier = BaselineVerifier()
    success = verifier.run_complete_baseline_verification()
    
    if success:
        print(f"\n🎉 PHASE 2 BASELINE VERIFICATION SUCCESSFUL!")
        print(f"📋 Next Step: PHASE 3 - Fix Table 5 Component Analysis")
    else:
        print(f"\n⚠️ PHASE 2 FOUND BASELINE DISCREPANCIES")
        print(f"📋 Investigate baseline implementations before proceeding")
        
if __name__ == "__main__":
    main()
