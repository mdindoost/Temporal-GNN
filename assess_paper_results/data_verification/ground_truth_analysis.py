#!/usr/bin/env python3
"""
PHASE 1: Data Foundation - Bitcoin Network Analysis and Ground Truth Establishment
Author: Paper Verification Team
Purpose: Establish exact dataset statistics and suspicious user identification

CRITICAL VERIFICATION TARGETS:
- 24,186 edges (paper claim)
- 3,783 users (paper claim) 
- 73 suspicious users (paper claim)
- >30% negative ratings with min 5 interactions (ground truth definition)
"""

import pandas as pd
import numpy as np
import gzip
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BitcoinDataVerifier:
    """
    Comprehensive Bitcoin network data verification and ground truth establishment
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
            'negative_threshold': 0.30,  # >30% negative ratings
            'min_interactions': 5
        }
        
        self.verification_results = {}
        self.data_stats = {}
        
    def load_bitcoin_networks(self):
        """Load and combine Bitcoin Alpha and OTC networks"""
        print("🔍 PHASE 1.1: Loading Bitcoin Networks...")
        
        # Load Bitcoin Alpha
        print(f"Loading Bitcoin Alpha from: {self.alpha_file}")
        try:
            alpha_df = pd.read_csv(self.alpha_file, names=['source', 'target', 'rating', 'timestamp'])
            print(f"✅ Bitcoin Alpha loaded: {len(alpha_df)} edges")
        except Exception as e:
            print(f"❌ Error loading Bitcoin Alpha: {e}")
            return None
            
        # Load Bitcoin OTC  
        print(f"Loading Bitcoin OTC from: {self.otc_file}")
        try:
            otc_df = pd.read_csv(self.otc_file, names=['source', 'target', 'rating', 'timestamp'])
            print(f"✅ Bitcoin OTC loaded: {len(otc_df)} edges")
        except Exception as e:
            print(f"❌ Error loading Bitcoin OTC: {e}")
            return None
            
        # Combine networks (paper methodology)
        combined_df = pd.concat([alpha_df, otc_df], ignore_index=True)
        print(f"✅ Combined network: {len(combined_df)} edges")
        
        # Add network source for tracking
        alpha_df['network'] = 'alpha'
        otc_df['network'] = 'otc'
        combined_df['network'] = 'combined'
        
        self.networks = {
            'alpha': alpha_df,
            'otc': otc_df, 
            'combined': combined_df
        }
        
        return True
        
    def verify_basic_statistics(self):
        """Verify basic network statistics against paper claims"""
        print("\n🔍 PHASE 1.2: Verifying Basic Statistics...")
        
        combined_df = self.networks['combined']
        
        # Calculate statistics
        total_edges = len(combined_df)
        unique_users = len(set(combined_df['source'].unique()) | set(combined_df['target'].unique()))
        
        # Rating distribution
        rating_dist = combined_df['rating'].value_counts().sort_index()
        negative_ratings = len(combined_df[combined_df['rating'] == -1])
        positive_ratings = len(combined_df[combined_df['rating'] == 1])
        
        stats = {
            'total_edges': total_edges,
            'total_users': unique_users,
            'negative_ratings': negative_ratings,
            'positive_ratings': positive_ratings,
            'negative_ratio': negative_ratings / total_edges,
            'rating_distribution': rating_dist.to_dict()
        }
        
        self.data_stats = stats
        
        # Verification against paper claims
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"Paper Claim: {self.paper_claims['total_edges']} edges")
        print(f"Actual Data: {total_edges} edges")
        print(f"✅ Match: {total_edges == self.paper_claims['total_edges']}")
        
        print(f"\nPaper Claim: {self.paper_claims['total_users']} users")
        print(f"Actual Data: {unique_users} users") 
        print(f"✅ Match: {unique_users == self.paper_claims['total_users']}")
        
        # Store verification results
        self.verification_results['basic_stats'] = {
            'edges_match': total_edges == self.paper_claims['total_edges'],
            'users_match': unique_users == self.paper_claims['total_users'],
            'actual_edges': total_edges,
            'actual_users': unique_users,
            'paper_edges': self.paper_claims['total_edges'],
            'paper_users': self.paper_claims['total_users']
        }
        
        return stats
        
    def identify_suspicious_users(self):
        """Identify suspicious users using paper methodology"""
        print("\n🔍 PHASE 1.3: Identifying Suspicious Users...")
        
        combined_df = self.networks['combined']
        
        # Calculate user statistics (as targets - receiving ratings)
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0, 'positive': 0, 'ratings': []})
        
        for _, row in combined_df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            user_stats[target]['ratings'].append(rating)
            
            if rating == -1:
                user_stats[target]['negative'] += 1
            elif rating == 1:
                user_stats[target]['positive'] += 1
                
        # Apply paper criteria: >30% negative ratings with min 5 interactions
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
                    
        # Sort by negative ratio for analysis
        user_analysis.sort(key=lambda x: x['negative_ratio'], reverse=True)
        
        print(f"\n📊 SUSPICIOUS USER ANALYSIS:")
        print(f"Users with ≥{self.paper_claims['min_interactions']} interactions: {len(user_analysis)}")
        print(f"Suspicious users (>{self.paper_claims['negative_threshold']*100}% negative): {len(suspicious_users)}")
        print(f"Paper claim: {self.paper_claims['suspicious_users']} suspicious users")
        print(f"✅ Match: {len(suspicious_users) == self.paper_claims['suspicious_users']}")
        
        # Show top suspicious users
        print(f"\n🔴 TOP 10 SUSPICIOUS USERS:")
        for i, user in enumerate(user_analysis[:10]):
            if user['is_suspicious']:
                print(f"{i+1:2d}. User {user['user_id']:4d}: {user['negative_ratio']:.3f} "
                      f"({user['negative_ratings']}/{user['total_interactions']})")
                      
        # Store results
        self.suspicious_users = suspicious_users
        self.user_analysis = user_analysis
        
        # Verification
        self.verification_results['suspicious_users'] = {
            'count_match': len(suspicious_users) == self.paper_claims['suspicious_users'],
            'actual_count': len(suspicious_users),
            'paper_count': self.paper_claims['suspicious_users'],
            'methodology_confirmed': True
        }
        
        return suspicious_users, user_analysis
        
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data"""
        print("\n🔍 PHASE 1.4: Temporal Pattern Analysis...")
        
        combined_df = self.networks['combined']
        
        # Convert timestamps
        combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='s')
        combined_df['year'] = combined_df['datetime'].dt.year
        combined_df['month'] = combined_df['datetime'].dt.month
        
        # Temporal statistics
        temporal_stats = {
            'date_range': {
                'start': combined_df['datetime'].min(),
                'end': combined_df['datetime'].max(),
                'span_days': (combined_df['datetime'].max() - combined_df['datetime'].min()).days
            },
            'yearly_distribution': combined_df['year'].value_counts().sort_index().to_dict(),
            'monthly_activity': combined_df.groupby(['year', 'month']).size().to_dict()
        }
        
        print(f"📅 TEMPORAL CHARACTERISTICS:")
        print(f"Date range: {temporal_stats['date_range']['start']} to {temporal_stats['date_range']['end']}")
        print(f"Span: {temporal_stats['date_range']['span_days']} days")
        print(f"Yearly distribution: {temporal_stats['yearly_distribution']}")
        
        self.temporal_stats = temporal_stats
        return temporal_stats
        
    def save_verification_results(self, output_dir="data_verification"):
        """Save all verification results and ground truth data"""
        print(f"\n💾 PHASE 1.5: Saving Verification Results to {output_dir}...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Complete verification report
        verification_report = {
            'timestamp': datetime.now().isoformat(),
            'paper_claims': self.paper_claims,
            'verification_results': self.verification_results,
            'data_statistics': self.data_stats,
            'temporal_analysis': self.temporal_stats,
            'suspicious_users_list': self.suspicious_users,
            'discrepancies_found': [],
            'status': 'VERIFIED' if all(
                result.get('edges_match', True) and 
                result.get('users_match', True) and 
                result.get('count_match', True)
                for result in self.verification_results.values()
            ) else 'DISCREPANCIES_FOUND'
        }
        
        # Check for discrepancies
        for category, results in self.verification_results.items():
            for key, value in results.items():
                if 'match' in key and not value:
                    verification_report['discrepancies_found'].append({
                        'category': category,
                        'issue': key,
                        'details': results
                    })
        
        # Save verification report
        with open(f"{output_dir}/verification_report.json", 'w') as f:
            json.dump(verification_report, f, indent=2, default=str)
            
        # Save ground truth suspicious users
        suspicious_df = pd.DataFrame(self.user_analysis)
        suspicious_df.to_csv(f"{output_dir}/ground_truth_analysis.csv", index=False)
        
        # Save network data with suspicious labels
        combined_df = self.networks['combined'].copy()
        combined_df['target_is_suspicious'] = combined_df['target'].isin(self.suspicious_users)
        combined_df['source_is_suspicious'] = combined_df['source'].isin(self.suspicious_users)
        combined_df.to_csv(f"{output_dir}/bitcoin_network_labeled.csv", index=False)
        
        print(f"✅ Saved: verification_report.json")
        print(f"✅ Saved: ground_truth_analysis.csv") 
        print(f"✅ Saved: bitcoin_network_labeled.csv")
        
        return verification_report
        
    def create_verification_visualizations(self, output_dir="../figures"):
        """Create verification visualizations"""
        print(f"\n📊 PHASE 1.6: Creating Verification Visualizations...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Figure 1: Data verification dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Network Data Verification Dashboard', fontsize=16, fontweight='bold')
        
        # Rating distribution
        combined_df = self.networks['combined']
        rating_counts = combined_df['rating'].value_counts().sort_index()
        axes[0,0].bar(rating_counts.index, rating_counts.values, color=['red', 'blue'])
        axes[0,0].set_title('Rating Distribution')
        axes[0,0].set_xlabel('Rating Value')
        axes[0,0].set_ylabel('Count')
        
        # Suspicious user threshold analysis
        user_df = pd.DataFrame(self.user_analysis)
        axes[0,1].hist(user_df['negative_ratio'], bins=20, alpha=0.7, color='orange')
        axes[0,1].axvline(x=0.30, color='red', linestyle='--', label='Threshold (30%)')
        axes[0,1].set_title('Negative Rating Ratio Distribution')
        axes[0,1].set_xlabel('Negative Rating Ratio')
        axes[0,1].set_ylabel('Number of Users')
        axes[0,1].legend()
        
        # Temporal activity
        yearly_dist = pd.Series(self.temporal_stats['yearly_distribution'])
        axes[1,0].bar(yearly_dist.index, yearly_dist.values, color='green')
        axes[1,0].set_title('Yearly Activity Distribution')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Number of Ratings')
        
        # Verification status
        verification_text = []
        for category, results in self.verification_results.items():
            for key, value in results.items():
                if 'match' in key:
                    status = "✅ PASS" if value else "❌ FAIL"
                    verification_text.append(f"{category}.{key}: {status}")
                    
        axes[1,1].text(0.1, 0.9, '\n'.join(verification_text), 
                      transform=axes[1,1].transAxes, fontsize=10,
                      verticalalignment='top', fontfamily='monospace')
        axes[1,1].set_title('Verification Status')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_verification_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: data_verification_dashboard.png")
        
    def run_complete_verification(self):
        """Run complete data verification pipeline"""
        print("🚀 STARTING PHASE 1: DATA FOUNDATION VERIFICATION")
        print("="*60)
        
        # Load data
        if not self.load_bitcoin_networks():
            return False
            
        # Verify statistics  
        self.verify_basic_statistics()
        
        # Identify suspicious users
        self.identify_suspicious_users()
        
        # Analyze temporal patterns
        self.analyze_temporal_patterns()
        
        # Save results
        report = self.save_verification_results()
        
        # Create visualizations
        self.create_verification_visualizations()
        
        # Final status
        print("\n" + "="*60)
        print(f"🎯 PHASE 1 COMPLETE - STATUS: {report['status']}")
        
        if report['discrepancies_found']:
            print("❌ DISCREPANCIES FOUND:")
            for disc in report['discrepancies_found']:
                print(f"   - {disc['category']}: {disc['issue']}")
        else:
            print("✅ ALL VERIFICATIONS PASSED")
            
        print(f"📁 Results saved to: data_verification/")
        print("🔄 Ready for PHASE 2: Baseline Implementation")
        
        return report['status'] == 'VERIFIED'

def main():
    """Main verification execution"""
    verifier = BitcoinDataVerifier()
    success = verifier.run_complete_verification()
    
    if success:
        print("\n🎉 PHASE 1 VERIFICATION SUCCESSFUL!")
        print("📋 Next Step: Run baseline verification (PHASE 2)")
    else:
        print("\n⚠️  PHASE 1 FOUND DISCREPANCIES - INVESTIGATE BEFORE PROCEEDING")
        
if __name__ == "__main__":
    main()
