#!/usr/bin/env python3
"""
TABLE 1 DEEP INVESTIGATION: Find exact evaluation methodology
Author: Paper Verification Team
Purpose: Investigate why Table 1 results don't match even with correct 73 suspicious users

CURRENT STATUS:
- ✅ Dataset: bitcoin_alpha_processed.csv (24,186 edges, 3,783 users)
- ✅ Suspicious users: 73 (negative ratio > 18%, min interactions ≥ 7)
- ❌ Table 1 results: Still don't match paper claims

HYPOTHESIS:
1. Different evaluation period (temporal subsets)
2. Different negative rating encoding (-1 vs different values)
3. Different separation ratio calculation
4. Different precision@50 calculation
5. Different dataset preprocessing
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

class Table1DeepInvestigator:
    """
    Deep investigation of Table 1 evaluation methodology discrepancies
    """
    
    def __init__(self):
        # Load exact dataset and ground truth
        self.df = pd.read_csv('/home/md724/temporal-gnn-project/data/processed/bitcoin_alpha_processed.csv')
        
        with open('data_verification/corrected_pipeline_results.json', 'r') as f:
            results = json.load(f)
            self.suspicious_users = results['suspicious_user_investigation']['suspicious_users_list']
            
        # Paper claims to investigate
        self.paper_claims = {
            'negative_ratio': {'separation_ratio': 25.08, 'precision_at_50': 0.460},
            'temporal_volatility': {'separation_ratio': 2.46, 'precision_at_50': 0.580},
            'weighted_pagerank': {'separation_ratio': 2.15, 'precision_at_50': 0.280}
        }
        
        print(f"🔍 Loaded dataset: {len(self.df)} edges")
        print(f"🔍 Suspicious users: {len(self.suspicious_users)}")
        
    def analyze_rating_distribution(self):
        """Analyze rating distribution to understand encoding"""
        print(f"\n📊 ANALYZING RATING DISTRIBUTION")
        print("="*40)
        
        rating_counts = self.df['rating'].value_counts().sort_index()
        print(f"Rating distribution:")
        for rating, count in rating_counts.items():
            print(f"   Rating {rating}: {count:,} edges ({count/len(self.df)*100:.1f}%)")
            
        # Check if ratings are encoded differently than expected
        unique_ratings = sorted(self.df['rating'].unique())
        print(f"\nUnique ratings: {unique_ratings}")
        
        # Negative ratings analysis
        negative_ratings = self.df[self.df['rating'] < 0]
        positive_ratings = self.df[self.df['rating'] > 0]
        
        print(f"Negative ratings: {len(negative_ratings):,} ({len(negative_ratings)/len(self.df)*100:.1f}%)")
        print(f"Positive ratings: {len(positive_ratings):,} ({len(positive_ratings)/len(self.df)*100:.1f}%)")
        
        return rating_counts, unique_ratings
        
    def test_different_separation_calculations(self):
        """Test different ways to calculate separation ratio"""
        print(f"\n🧪 TESTING DIFFERENT SEPARATION RATIO CALCULATIONS")
        print("="*55)
        
        # Calculate user negative ratios
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0, 'positive': 0})
        
        for _, row in self.df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            if rating < 0:
                user_stats[target]['negative'] += 1
            elif rating > 0:
                user_stats[target]['positive'] += 1
                
        # Calculate negative ratios
        user_negative_ratios = {}
        for user_id, stats in user_stats.items():
            if stats['total'] > 0:
                negative_ratio = stats['negative'] / stats['total']
                user_negative_ratios[user_id] = negative_ratio
            else:
                user_negative_ratios[user_id] = 0.0
                
        # Test different separation calculations
        separation_methods = [
            ("Mean Suspicious / Mean Normal", self.calc_mean_separation),
            ("Median Suspicious / Median Normal", self.calc_median_separation),
            ("Max Suspicious / Mean Normal", self.calc_max_mean_separation),
            ("Mean Suspicious / Min Normal", self.calc_mean_min_separation),
            ("Percentile-based (95th vs 5th)", self.calc_percentile_separation),
            ("Weighted by interactions", self.calc_weighted_separation)
        ]
        
        results = {}
        
        for method_name, calc_func in separation_methods:
            try:
                separation = calc_func(user_negative_ratios, user_stats)
                results[method_name] = separation
                
                match_25 = abs(separation - 25.08) < 2.0
                print(f"   {method_name:<35}: {separation:6.2f} {'🎯' if match_25 else ''}")
                
            except Exception as e:
                print(f"   {method_name:<35}: ERROR - {e}")
                results[method_name] = None
                
        return results
        
    def calc_mean_separation(self, user_negative_ratios, user_stats):
        """Standard mean separation calculation"""
        suspicious_ratios = [user_negative_ratios[user] for user in self.suspicious_users 
                           if user in user_negative_ratios]
        normal_ratios = [user_negative_ratios[user] for user in user_negative_ratios 
                        if user not in self.suspicious_users]
        
        if len(suspicious_ratios) > 0 and len(normal_ratios) > 0:
            return np.mean(suspicious_ratios) / np.mean(normal_ratios)
        return 0.0
        
    def calc_median_separation(self, user_negative_ratios, user_stats):
        """Median-based separation"""
        suspicious_ratios = [user_negative_ratios[user] for user in self.suspicious_users 
                           if user in user_negative_ratios]
        normal_ratios = [user_negative_ratios[user] for user in user_negative_ratios 
                        if user not in self.suspicious_users]
        
        if len(suspicious_ratios) > 0 and len(normal_ratios) > 0:
            return np.median(suspicious_ratios) / np.median(normal_ratios)
        return 0.0
        
    def calc_max_mean_separation(self, user_negative_ratios, user_stats):
        """Max suspicious vs mean normal"""
        suspicious_ratios = [user_negative_ratios[user] for user in self.suspicious_users 
                           if user in user_negative_ratios]
        normal_ratios = [user_negative_ratios[user] for user in user_negative_ratios 
                        if user not in self.suspicious_users]
        
        if len(suspicious_ratios) > 0 and len(normal_ratios) > 0:
            return np.max(suspicious_ratios) / np.mean(normal_ratios)
        return 0.0
        
    def calc_mean_min_separation(self, user_negative_ratios, user_stats):
        """Mean suspicious vs min normal"""
        suspicious_ratios = [user_negative_ratios[user] for user in self.suspicious_users 
                           if user in user_negative_ratios]
        normal_ratios = [user_negative_ratios[user] for user in user_negative_ratios 
                        if user not in self.suspicious_users and user_negative_ratios[user] > 0]
        
        if len(suspicious_ratios) > 0 and len(normal_ratios) > 0:
            return np.mean(suspicious_ratios) / np.min(normal_ratios)
        return 0.0
        
    def calc_percentile_separation(self, user_negative_ratios, user_stats):
        """95th percentile suspicious vs 5th percentile normal"""
        suspicious_ratios = [user_negative_ratios[user] for user in self.suspicious_users 
                           if user in user_negative_ratios]
        normal_ratios = [user_negative_ratios[user] for user in user_negative_ratios 
                        if user not in self.suspicious_users]
        
        if len(suspicious_ratios) > 0 and len(normal_ratios) > 0:
            susp_95 = np.percentile(suspicious_ratios, 95)
            norm_5 = np.percentile(normal_ratios, 5)
            return susp_95 / norm_5 if norm_5 > 0 else float('inf')
        return 0.0
        
    def calc_weighted_separation(self, user_negative_ratios, user_stats):
        """Weighted by number of interactions"""
        suspicious_weighted = 0
        suspicious_total_weight = 0
        normal_weighted = 0
        normal_total_weight = 0
        
        for user_id, ratio in user_negative_ratios.items():
            weight = user_stats[user_id]['total']
            
            if user_id in self.suspicious_users:
                suspicious_weighted += ratio * weight
                suspicious_total_weight += weight
            else:
                normal_weighted += ratio * weight
                normal_total_weight += weight
                
        if suspicious_total_weight > 0 and normal_total_weight > 0:
            avg_suspicious = suspicious_weighted / suspicious_total_weight
            avg_normal = normal_weighted / normal_total_weight
            return avg_suspicious / avg_normal if avg_normal > 0 else float('inf')
        return 0.0
        
    def test_different_precision_calculations(self):
        """Test different precision@50 calculations"""
        print(f"\n🧪 TESTING DIFFERENT PRECISION@50 CALCULATIONS")
        print("="*50)
        
        # Calculate user negative ratios
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0})
        
        for _, row in self.df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            if rating < 0:
                user_stats[target]['negative'] += 1
                
        user_negative_ratios = {}
        for user_id, stats in user_stats.items():
            if stats['total'] > 0:
                negative_ratio = stats['negative'] / stats['total']
                user_negative_ratios[user_id] = negative_ratio
                
        # Test different precision calculations
        precision_methods = [
            ("Standard Top 50", self.calc_standard_precision),
            ("Filtered by min interactions", self.calc_filtered_precision),
            ("Weighted by interactions", self.calc_weighted_precision),
            ("Top 50 among active users", self.calc_active_users_precision)
        ]
        
        results = {}
        
        for method_name, calc_func in precision_methods:
            try:
                precision = calc_func(user_negative_ratios, user_stats)
                results[method_name] = precision
                
                match_46 = abs(precision - 0.460) < 0.1
                print(f"   {method_name:<30}: {precision:.3f} {'🎯' if match_46 else ''}")
                
            except Exception as e:
                print(f"   {method_name:<30}: ERROR - {e}")
                results[method_name] = None
                
        return results
        
    def calc_standard_precision(self, user_negative_ratios, user_stats):
        """Standard precision@50 calculation"""
        sorted_users = sorted(user_negative_ratios.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:50]]
        
        # Create labels
        user_labels = {user: 1 if user in self.suspicious_users else 0 
                      for user in user_negative_ratios.keys()}
        
        top_50_labels = [user_labels[user] for user in top_50_users]
        return sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
    def calc_filtered_precision(self, user_negative_ratios, user_stats):
        """Precision@50 among users with ≥5 interactions"""
        # Filter users with minimum interactions
        filtered_users = {user_id: ratio for user_id, ratio in user_negative_ratios.items()
                         if user_stats[user_id]['total'] >= 5}
        
        sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:50]]
        
        user_labels = {user: 1 if user in self.suspicious_users else 0 
                      for user in filtered_users.keys()}
        
        top_50_labels = [user_labels[user] for user in top_50_users]
        return sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
    def calc_weighted_precision(self, user_negative_ratios, user_stats):
        """Weighted precision@50"""
        # Weight scores by number of interactions
        weighted_scores = {}
        for user_id, ratio in user_negative_ratios.items():
            weight = user_stats[user_id]['total']
            weighted_scores[user_id] = ratio * np.log(1 + weight)  # Log weighting
            
        sorted_users = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:50]]
        
        user_labels = {user: 1 if user in self.suspicious_users else 0 
                      for user in weighted_scores.keys()}
        
        top_50_labels = [user_labels[user] for user in top_50_users]
        return sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
    def calc_active_users_precision(self, user_negative_ratios, user_stats):
        """Precision@50 among users with ≥10 interactions"""
        # Filter for active users
        active_users = {user_id: ratio for user_id, ratio in user_negative_ratios.items()
                       if user_stats[user_id]['total'] >= 10}
        
        sorted_users = sorted(active_users.items(), key=lambda x: x[1], reverse=True)
        top_50_users = [user_id for user_id, score in sorted_users[:min(50, len(sorted_users))]]
        
        user_labels = {user: 1 if user in self.suspicious_users else 0 
                      for user in active_users.keys()}
        
        top_50_labels = [user_labels[user] for user in top_50_users]
        return sum(top_50_labels) / len(top_50_labels) if top_50_labels else 0.0
        
    def investigate_temporal_subsets(self):
        """Test if paper used temporal subsets of data"""
        print(f"\n🧪 TESTING TEMPORAL SUBSETS")
        print("="*35)
        
        # Convert timestamps to datetime
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
        
        # Get date range
        start_date = self.df['datetime'].min()
        end_date = self.df['datetime'].max()
        total_days = (end_date - start_date).days
        
        print(f"Data range: {start_date.date()} to {end_date.date()} ({total_days} days)")
        
        # Test different temporal windows
        temporal_windows = [
            ("First 50%", 0.0, 0.5),
            ("Last 50%", 0.5, 1.0),
            ("First 25%", 0.0, 0.25),
            ("Last 25%", 0.75, 1.0),
            ("Middle 50%", 0.25, 0.75),
            ("First year", 0.0, 365/total_days),
            ("Last year", 1.0 - 365/total_days, 1.0)
        ]
        
        print(f"\nTesting temporal windows:")
        
        for window_name, start_frac, end_frac in temporal_windows:
            try:
                window_start = start_date + pd.Timedelta(days=total_days * start_frac)
                window_end = start_date + pd.Timedelta(days=total_days * end_frac)
                
                # Filter data to window
                window_df = self.df[
                    (self.df['datetime'] >= window_start) & 
                    (self.df['datetime'] <= window_end)
                ]
                
                if len(window_df) > 1000:  # Only test if reasonable amount of data
                    # Quick negative ratio calculation
                    separation = self.calc_window_separation(window_df)
                    
                    match_25 = abs(separation - 25.08) < 3.0
                    print(f"   {window_name:<15}: {len(window_df):5d} edges, separation: {separation:6.2f} {'🎯' if match_25 else ''}")
                    
            except Exception as e:
                print(f"   {window_name:<15}: ERROR - {e}")
                
    def calc_window_separation(self, window_df):
        """Calculate separation ratio for a temporal window"""
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0})
        
        for _, row in window_df.iterrows():
            target = row['target']
            rating = row['rating']
            
            user_stats[target]['total'] += 1
            if rating < 0:
                user_stats[target]['negative'] += 1
                
        user_negative_ratios = {}
        for user_id, stats in user_stats.items():
            if stats['total'] > 0:
                negative_ratio = stats['negative'] / stats['total']
                user_negative_ratios[user_id] = negative_ratio
                
        # Calculate separation
        suspicious_ratios = [user_negative_ratios[user] for user in self.suspicious_users 
                           if user in user_negative_ratios]
        normal_ratios = [user_negative_ratios[user] for user in user_negative_ratios 
                        if user not in self.suspicious_users]
        
        if len(suspicious_ratios) > 0 and len(normal_ratios) > 0:
            return np.mean(suspicious_ratios) / np.mean(normal_ratios)
        return 0.0
        
    def run_complete_investigation(self):
        """Run complete Table 1 investigation"""
        print("🚀 STARTING TABLE 1 DEEP INVESTIGATION")
        print("="*45)
        
        # Analyze rating distribution
        self.analyze_rating_distribution()
        
        # Test separation calculations
        sep_results = self.test_different_separation_calculations()
        
        # Test precision calculations
        prec_results = self.test_different_precision_calculations()
        
        # Test temporal subsets
        self.investigate_temporal_subsets()
        
        # Summary
        print(f"\n🎯 INVESTIGATION SUMMARY")
        print("="*25)
        
        # Find closest matches
        best_separation = None
        best_sep_diff = float('inf')
        
        for method, value in sep_results.items():
            if value is not None:
                diff = abs(value - 25.08)
                if diff < best_sep_diff:
                    best_sep_diff = diff
                    best_separation = (method, value)
                    
        best_precision = None
        best_prec_diff = float('inf')
        
        for method, value in prec_results.items():
            if value is not None:
                diff = abs(value - 0.460)
                if diff < best_prec_diff:
                    best_prec_diff = diff
                    best_precision = (method, value)
                    
        if best_separation:
            print(f"✅ Best separation match: {best_separation[0]}")
            print(f"   Value: {best_separation[1]:.2f} (paper: 25.08, diff: {best_sep_diff:.2f})")
            
        if best_precision:
            print(f"✅ Best precision match: {best_precision[0]}")
            print(f"   Value: {best_precision[1]:.3f} (paper: 0.460, diff: {best_prec_diff:.3f})")
            
        # Decision
        if best_sep_diff < 5.0 and best_prec_diff < 0.2:
            print(f"\n🎯 LIKELY FOUND PAPER'S METHODOLOGY!")
            print(f"📋 Proceed to Table 5 with identified evaluation approach")
        else:
            print(f"\n⚠️ METHODOLOGY STILL UNCLEAR")
            print(f"📋 Consider using current best approximation for Table 5 verification")

def main():
    """Main Table 1 investigation"""
    investigator = Table1DeepInvestigator()
    investigator.run_complete_investigation()

if __name__ == "__main__":
    main()
