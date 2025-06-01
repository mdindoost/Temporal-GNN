#!/usr/bin/env python3
"""
THRESHOLD ANALYSIS: Find criteria that yield 73 suspicious users
Author: Paper Verification Team
Purpose: Investigate what threshold/criteria produces paper's claim of 73 suspicious users

DISCOVERED ISSUE:
- Bitcoin Alpha has 24,186 edges ✅ and 3,783 users ✅ 
- But only 9 suspicious users with >30% negative, min 5 interactions
- Paper claims 73 suspicious users - need to find correct criteria
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class ThresholdAnalyzer:
    """
    Analyze different threshold combinations to match paper's 73 suspicious users
    """
    
    def __init__(self):
        self.alpha_file = "/home/md724/temporal-gnn-project/data/bitcoin/soc-sign-bitcoin-alpha.csv.gz"
        self.df = None
        
    def load_bitcoin_alpha(self):
        """Load Bitcoin Alpha dataset"""
        print("📁 Loading Bitcoin Alpha dataset...")
        self.df = pd.read_csv(self.alpha_file, names=['source', 'target', 'rating', 'timestamp'])
        print(f"✅ Loaded: {len(self.df)} edges, {len(set(self.df['source']) | set(self.df['target']))} users")
        return True
        
    def analyze_user_statistics(self):
        """Comprehensive analysis of user interaction patterns"""
        print("\n🔍 ANALYZING USER INTERACTION PATTERNS...")
        
        # Calculate comprehensive user statistics
        user_stats = defaultdict(lambda: {
            'total': 0, 'negative': 0, 'positive': 0, 'ratings': []
        })
        
        for _, row in self.df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            user_stats[target]['ratings'].append(rating)
            
            if rating == -1:
                user_stats[target]['negative'] += 1
            elif rating == 1:
                user_stats[target]['positive'] += 1
                
        # Convert to DataFrame for analysis
        user_analysis = []
        for user_id, stats in user_stats.items():
            if stats['total'] > 0:
                negative_ratio = stats['negative'] / stats['total']
                user_analysis.append({
                    'user_id': user_id,
                    'total_interactions': stats['total'],
                    'negative_ratings': stats['negative'],
                    'positive_ratings': stats['positive'],
                    'negative_ratio': negative_ratio
                })
                
        user_df = pd.DataFrame(user_analysis)
        user_df = user_df.sort_values('negative_ratio', ascending=False)
        
        print(f"📊 USER STATISTICS SUMMARY:")
        print(f"   Total users with ratings: {len(user_df)}")
        print(f"   Users with ≥1 interactions: {len(user_df[user_df['total_interactions'] >= 1])}")
        print(f"   Users with ≥2 interactions: {len(user_df[user_df['total_interactions'] >= 2])}")
        print(f"   Users with ≥3 interactions: {len(user_df[user_df['total_interactions'] >= 3])}")
        print(f"   Users with ≥5 interactions: {len(user_df[user_df['total_interactions'] >= 5])}")
        print(f"   Users with ≥10 interactions: {len(user_df[user_df['total_interactions'] >= 10])}")
        
        self.user_df = user_df
        return user_df
        
    def test_threshold_combinations(self):
        """Test different threshold combinations to find 73 suspicious users"""
        print("\n🧪 TESTING THRESHOLD COMBINATIONS TO FIND 73 SUSPICIOUS USERS...")
        
        # Test various combinations
        test_scenarios = [
            # (negative_threshold, min_interactions, description)
            (0.30, 5, "Paper claimed criteria"),
            (0.25, 5, "Lower negative threshold"),
            (0.20, 5, "Much lower negative threshold"),
            (0.15, 5, "Very low negative threshold"),
            (0.10, 5, "Extremely low negative threshold"),
            (0.30, 3, "Lower interaction minimum"),
            (0.30, 2, "Very low interaction minimum"),
            (0.30, 1, "Any interaction"),
            (0.25, 3, "Combined: lower threshold + interactions"),
            (0.20, 3, "More relaxed criteria"),
            (0.15, 3, "Very relaxed criteria"),
            (0.10, 3, "Extremely relaxed criteria"),
            (0.05, 2, "Ultra relaxed criteria"),
            (0.01, 1, "Almost any negative rating"),
        ]
        
        results = []
        
        print(f"\n{'Threshold':<12} {'Min Int':<8} {'Count':<8} {'Description':<25} {'Match':<8}")
        print("-" * 70)
        
        for neg_threshold, min_interactions, description in test_scenarios:
            # Filter users by minimum interactions
            filtered_users = self.user_df[
                self.user_df['total_interactions'] >= min_interactions
            ]
            
            # Count suspicious users
            suspicious_users = filtered_users[
                filtered_users['negative_ratio'] > neg_threshold
            ]
            
            count = len(suspicious_users)
            is_match = count == 73
            match_indicator = "🎯 YES" if is_match else ""
            
            print(f"{neg_threshold:<12.2f} {min_interactions:<8} {count:<8} {description:<25} {match_indicator}")
            
            results.append({
                'negative_threshold': neg_threshold,
                'min_interactions': min_interactions,
                'suspicious_count': count,
                'description': description,
                'is_match': is_match,
                'suspicious_users': suspicious_users['user_id'].tolist() if is_match else []
            })
            
        # Find exact matches
        exact_matches = [r for r in results if r['is_match']]
        
        if exact_matches:
            print(f"\n🎯 FOUND EXACT MATCHES:")
            for match in exact_matches:
                print(f"   Threshold: {match['negative_threshold']:.2f}")
                print(f"   Min interactions: {match['min_interactions']}")
                print(f"   Description: {match['description']}")
        else:
            print(f"\n❌ NO EXACT MATCHES FOUND")
            # Find closest matches
            closest = min(results, key=lambda x: abs(x['suspicious_count'] - 73))
            print(f"   Closest: {closest['suspicious_count']} users with {closest['description']}")
            
        return results, exact_matches
        
    def analyze_negative_ratio_distribution(self):
        """Analyze the distribution of negative ratios to understand thresholds"""
        print("\n📊 ANALYZING NEGATIVE RATIO DISTRIBUTION...")
        
        # Get users with different minimum interactions
        interaction_thresholds = [1, 2, 3, 5, 10]
        
        for min_int in interaction_thresholds:
            filtered_users = self.user_df[self.user_df['total_interactions'] >= min_int]
            
            if len(filtered_users) == 0:
                continue
                
            print(f"\n📈 Users with ≥{min_int} interactions: {len(filtered_users)}")
            
            # Percentile analysis
            percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 99]
            print("   Negative ratio percentiles:")
            for p in percentiles:
                value = np.percentile(filtered_users['negative_ratio'], p)
                count = len(filtered_users[filtered_users['negative_ratio'] >= value])
                print(f"      {p}th percentile: {value:.3f} ({count} users)")
                
                # Check if this gives us 73 users
                if 70 <= count <= 76:  # Close to 73
                    print(f"         🎯 CLOSE TO 73: {count} users at {value:.3f} threshold")
                    
    def create_threshold_visualization(self):
        """Create visualization of threshold analysis"""
        print("\n📊 Creating threshold analysis visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Alpha: Suspicious User Threshold Analysis', fontsize=16, fontweight='bold')
        
        # 1. Negative ratio distribution
        axes[0,0].hist(self.user_df['negative_ratio'], bins=50, alpha=0.7, color='blue')
        axes[0,0].axvline(x=0.30, color='red', linestyle='--', label='Paper threshold (30%)')
        axes[0,0].set_title('Negative Ratio Distribution (All Users)')
        axes[0,0].set_xlabel('Negative Rating Ratio')
        axes[0,0].set_ylabel('Number of Users')
        axes[0,0].legend()
        
        # 2. Interaction count distribution
        axes[0,1].hist(self.user_df['total_interactions'], bins=50, alpha=0.7, color='green')
        axes[0,1].axvline(x=5, color='red', linestyle='--', label='Paper minimum (5)')
        axes[0,1].set_title('Interaction Count Distribution')
        axes[0,1].set_xlabel('Total Interactions')
        axes[0,1].set_ylabel('Number of Users')
        axes[0,1].legend()
        axes[0,1].set_xlim(0, 50)  # Focus on reasonable range
        
        # 3. Suspicious user count vs threshold
        thresholds = np.arange(0.01, 0.51, 0.01)
        counts_min5 = []
        counts_min3 = []
        
        for threshold in thresholds:
            count5 = len(self.user_df[
                (self.user_df['total_interactions'] >= 5) & 
                (self.user_df['negative_ratio'] > threshold)
            ])
            count3 = len(self.user_df[
                (self.user_df['total_interactions'] >= 3) & 
                (self.user_df['negative_ratio'] > threshold)
            ])
            counts_min5.append(count5)
            counts_min3.append(count3)
            
        axes[1,0].plot(thresholds, counts_min5, label='Min 5 interactions', color='blue')
        axes[1,0].plot(thresholds, counts_min3, label='Min 3 interactions', color='green')
        axes[1,0].axhline(y=73, color='red', linestyle='--', label='Target: 73 users')
        axes[1,0].axvline(x=0.30, color='red', linestyle=':', alpha=0.5, label='Paper threshold')
        axes[1,0].set_title('Suspicious User Count vs Threshold')
        axes[1,0].set_xlabel('Negative Ratio Threshold')
        axes[1,0].set_ylabel('Number of Suspicious Users')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Top suspicious users
        top_users = self.user_df.head(20)
        y_pos = np.arange(len(top_users))
        axes[1,1].barh(y_pos, top_users['negative_ratio'], color='red', alpha=0.7)
        axes[1,1].set_yticks(y_pos)
        axes[1,1].set_yticklabels([f"User {int(uid)}" for uid in top_users['user_id']])
        axes[1,1].set_title('Top 20 Users by Negative Ratio')
        axes[1,1].set_xlabel('Negative Rating Ratio')
        axes[1,1].axvline(x=0.30, color='blue', linestyle='--', label='Paper threshold')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('figures/threshold_analysis_bitcoin_alpha.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved: threshold_analysis_bitcoin_alpha.png")
        
    def run_complete_analysis(self):
        """Run complete threshold analysis"""
        print("🚀 STARTING THRESHOLD ANALYSIS FOR 73 SUSPICIOUS USERS")
        print("="*60)
        
        # Load data
        self.load_bitcoin_alpha()
        
        # Analyze user patterns
        self.analyze_user_statistics()
        
        # Test threshold combinations
        results, exact_matches = self.test_threshold_combinations()
        
        # Analyze distribution
        self.analyze_negative_ratio_distribution()
        
        # Create visualization
        self.create_threshold_visualization()
        
        # Final conclusion
        print(f"\n🎯 THRESHOLD ANALYSIS CONCLUSION:")
        print("="*60)
        
        if exact_matches:
            print(f"✅ FOUND CRITERIA THAT YIELD 73 SUSPICIOUS USERS:")
            for match in exact_matches:
                print(f"   • Negative ratio > {match['negative_threshold']:.2f}")
                print(f"   • Minimum interactions ≥ {match['min_interactions']}")
                print(f"   • {match['description']}")
        else:
            print(f"❌ NO CRITERIA FOUND THAT YIELD EXACTLY 73 SUSPICIOUS USERS")
            print(f"   Paper's claim of 73 suspicious users may be incorrect")
            print(f"   OR different preprocessing/criteria were used")
            
        print(f"\n📋 RECOMMENDATION:")
        if exact_matches:
            print(f"   Use criteria: negative_ratio > {exact_matches[0]['negative_threshold']:.2f}, min_interactions ≥ {exact_matches[0]['min_interactions']}")
            print(f"   Proceed with {exact_matches[0]['suspicious_count']} suspicious users")
        else:
            print(f"   Use Bitcoin Alpha with criteria that match data reality")
            print(f"   Consider using 9 suspicious users (>30% negative, ≥5 interactions)")
            print(f"   OR investigate paper's exact preprocessing steps")

def main():
    analyzer = ThresholdAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
