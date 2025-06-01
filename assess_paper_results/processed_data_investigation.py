#!/usr/bin/env python3
"""
PROCESSED DATA INVESTIGATION: Find the exact dataset used in the paper
Author: Paper Verification Team
Purpose: Analyze processed data files to identify correct dataset and ground truth

DISCOVERED FILES:
- bitcoin_alpha_processed.csv
- bitcoin_otc_processed.csv  
- bitcoin_alpha_node_mapping.csv
- bitcoin_otc_node_mapping.csv

PAPER CLAIMS TO MATCH:
- 24,186 edges
- 3,783 users
- 73 suspicious users
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from datetime import datetime

class ProcessedDataInvestigator:
    """
    Investigate processed data files to find exact paper dataset
    """
    
    def __init__(self):
        self.data_dir = "/home/md724/temporal-gnn-project/data/processed"
        self.paper_claims = {
            'total_edges': 24186,
            'total_users': 3783,
            'suspicious_users': 73
        }
        
    def analyze_all_processed_files(self):
        """Analyze all processed data files"""
        print("🔍 INVESTIGATING PROCESSED DATA FILES")
        print("="*50)
        
        files_to_check = [
            'bitcoin_alpha_processed.csv',
            'bitcoin_otc_processed.csv',
            'bitcoin_alpha_node_mapping.csv', 
            'bitcoin_otc_node_mapping.csv'
        ]
        
        file_analysis = {}
        
        for filename in files_to_check:
            filepath = f"{self.data_dir}/{filename}"
            print(f"\n📁 ANALYZING: {filename}")
            print("-" * 40)
            
            try:
                df = pd.read_csv(filepath)
                
                # Basic info
                print(f"✅ Loaded successfully")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                
                # Show first few rows
                print(f"   First 3 rows:")
                print(df.head(3).to_string())
                
                # Detailed analysis based on file type
                if 'processed' in filename and df.shape[1] >= 3:
                    analysis = self.analyze_edge_data(df, filename)
                elif 'mapping' in filename:
                    analysis = self.analyze_mapping_data(df, filename)
                else:
                    analysis = self.analyze_generic_data(df, filename)
                    
                file_analysis[filename] = analysis
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                file_analysis[filename] = {'error': str(e)}
                
        return file_analysis
        
    def analyze_edge_data(self, df, filename):
        """Analyze edge/interaction data"""
        print(f"\n🔍 EDGE DATA ANALYSIS:")
        
        # Try to identify edge columns
        possible_edge_cols = ['source', 'target', 'rating', 'timestamp', 'weight', 'from', 'to']
        edge_cols = [col for col in df.columns if col.lower() in possible_edge_cols]
        
        total_edges = len(df)
        
        # Try to count unique users
        unique_users = 0
        if len(edge_cols) >= 2:
            source_col = edge_cols[0]
            target_col = edge_cols[1] if len(edge_cols) > 1 else edge_cols[0]
            
            unique_users = len(set(df[source_col].unique()) | set(df[target_col].unique()))
            
        print(f"   Total edges: {total_edges}")
        print(f"   Unique users: {unique_users}")
        
        # Check against paper claims
        edges_match = total_edges == self.paper_claims['total_edges']
        users_match = unique_users == self.paper_claims['total_users']
        
        print(f"   📊 PAPER MATCH CHECK:")
        print(f"      Edges {total_edges} vs {self.paper_claims['total_edges']}: {'✅' if edges_match else '❌'}")
        print(f"      Users {unique_users} vs {self.paper_claims['total_users']}: {'✅' if users_match else '❌'}")
        
        # If this looks like the right dataset, analyze suspicious users
        suspicious_users = 0
        if edges_match and users_match:
            print(f"   🎯 POTENTIAL MATCH FOUND! Analyzing suspicious users...")
            suspicious_users = self.find_suspicious_users_in_processed(df)
            
        return {
            'type': 'edge_data',
            'total_edges': total_edges,
            'unique_users': unique_users,
            'suspicious_users': suspicious_users,
            'edges_match': edges_match,
            'users_match': users_match,
            'potential_paper_dataset': edges_match and users_match,
            'columns': list(df.columns)
        }
        
    def find_suspicious_users_in_processed(self, df):
        """Find suspicious users in processed data"""
        print(f"      🔍 Finding suspicious users...")
        
        # Try different column name possibilities
        possible_cols = {
            'source': ['source', 'from', 'user1', 'src'],
            'target': ['target', 'to', 'user2', 'dst', 'dest'],
            'rating': ['rating', 'weight', 'score', 'value']
        }
        
        # Find actual column names
        source_col = None
        target_col = None
        rating_col = None
        
        for col_type, possibilities in possible_cols.items():
            for col_name in possibilities:
                if col_name in df.columns:
                    if col_type == 'source':
                        source_col = col_name
                    elif col_type == 'target':
                        target_col = col_name
                    elif col_type == 'rating':
                        rating_col = col_name
                    break
                    
        if not all([source_col, target_col, rating_col]):
            print(f"      ❌ Could not identify source/target/rating columns")
            return 0
            
        print(f"      ✅ Using columns: source={source_col}, target={target_col}, rating={rating_col}")
        
        # Calculate user statistics
        user_stats = defaultdict(lambda: {'total': 0, 'negative': 0})
        
        for _, row in df.iterrows():
            target = row[target_col]
            rating = row[rating_col]
            
            user_stats[target]['total'] += 1
            if rating == -1:
                user_stats[target]['negative'] += 1
                
        # Test different thresholds to find 73 suspicious users
        thresholds_to_test = [
            (0.30, 5), (0.25, 5), (0.20, 5), (0.15, 5), (0.10, 5),
            (0.30, 3), (0.25, 3), (0.20, 3), (0.15, 3), (0.10, 3),
            (0.30, 2), (0.25, 2), (0.20, 2), (0.15, 2), (0.10, 2),
            (0.30, 1), (0.25, 1), (0.20, 1), (0.15, 1), (0.10, 1)
        ]
        
        print(f"      🧪 Testing thresholds to find 73 suspicious users:")
        for neg_threshold, min_interactions in thresholds_to_test:
            suspicious_count = 0
            for user_id, stats in user_stats.items():
                if stats['total'] >= min_interactions:
                    negative_ratio = stats['negative'] / stats['total']
                    if negative_ratio > neg_threshold:
                        suspicious_count += 1
                        
            if suspicious_count == 73:
                print(f"         🎯 FOUND: {neg_threshold:.2f} threshold, {min_interactions} min interactions = 73 users")
                return suspicious_count
            elif 70 <= suspicious_count <= 76:
                print(f"         🔶 CLOSE: {neg_threshold:.2f} threshold, {min_interactions} min interactions = {suspicious_count} users")
                
        # If no exact match, return the count with standard criteria
        suspicious_count = 0
        for user_id, stats in user_stats.items():
            if stats['total'] >= 5:
                negative_ratio = stats['negative'] / stats['total']
                if negative_ratio > 0.30:
                    suspicious_count += 1
                    
        print(f"      📊 Standard criteria (>30%, ≥5 interactions): {suspicious_count} suspicious users")
        return suspicious_count
        
    def analyze_mapping_data(self, df, filename):
        """Analyze node mapping data"""
        print(f"\n🔍 MAPPING DATA ANALYSIS:")
        print(f"   Shape: {df.shape}")
        
        return {
            'type': 'mapping_data',
            'total_mappings': len(df),
            'columns': list(df.columns)
        }
        
    def analyze_generic_data(self, df, filename):
        """Analyze generic data"""
        print(f"\n🔍 GENERIC DATA ANALYSIS:")
        print(f"   Shape: {df.shape}")
        
        return {
            'type': 'generic_data',
            'shape': df.shape,
            'columns': list(df.columns)
        }
        
    def create_processed_data_summary(self, file_analysis):
        """Create summary of processed data investigation"""
        print(f"\n🎯 PROCESSED DATA INVESTIGATION SUMMARY")
        print("="*50)
        
        potential_matches = []
        
        for filename, analysis in file_analysis.items():
            if analysis.get('potential_paper_dataset', False):
                potential_matches.append((filename, analysis))
                
        if potential_matches:
            print(f"✅ POTENTIAL PAPER DATASET(S) FOUND:")
            for filename, analysis in potential_matches:
                print(f"   📁 {filename}")
                print(f"      Edges: {analysis['total_edges']} ({'✅' if analysis['edges_match'] else '❌'})")
                print(f"      Users: {analysis['unique_users']} ({'✅' if analysis['users_match'] else '❌'})")
                print(f"      Suspicious: {analysis['suspicious_users']}")
                if analysis['suspicious_users'] == 73:
                    print(f"      🎯 PERFECT MATCH - This is likely the paper's dataset!")
                    
        else:
            print(f"❌ NO PERFECT MATCHES FOUND")
            print(f"   Need to investigate data preprocessing pipeline")
            
        return potential_matches
        
    def save_processed_investigation_results(self, file_analysis, potential_matches):
        """Save investigation results"""
        print(f"\n💾 Saving Processed Data Investigation Results...")
        
        investigation_report = {
            'timestamp': datetime.now().isoformat(),
            'investigation_type': 'PROCESSED_DATA_ANALYSIS',
            'paper_claims': self.paper_claims,
            'file_analysis': {},
            'potential_matches': [],
            'recommendation': ''
        }
        
        # Convert file analysis to serializable format
        for filename, analysis in file_analysis.items():
            if 'error' not in analysis:
                investigation_report['file_analysis'][filename] = {
                    'type': analysis.get('type', 'unknown'),
                    'total_edges': analysis.get('total_edges', 0),
                    'unique_users': analysis.get('unique_users', 0),
                    'suspicious_users': analysis.get('suspicious_users', 0),
                    'edges_match': analysis.get('edges_match', False),
                    'users_match': analysis.get('users_match', False),
                    'potential_paper_dataset': analysis.get('potential_paper_dataset', False),
                    'columns': analysis.get('columns', [])
                }
            else:
                investigation_report['file_analysis'][filename] = {'error': analysis['error']}
                
        # Add potential matches
        for filename, analysis in potential_matches:
            investigation_report['potential_matches'].append({
                'filename': filename,
                'edges': analysis['total_edges'],
                'users': analysis['unique_users'],
                'suspicious_users': analysis['suspicious_users']
            })
            
        # Recommendation
        if potential_matches:
            best_match = max(potential_matches, key=lambda x: x[1]['suspicious_users'] == 73)
            investigation_report['recommendation'] = f"Use {best_match[0]} as paper dataset"
        else:
            investigation_report['recommendation'] = "Investigate data preprocessing pipeline"
            
        # Save report
        with open('/home/md724/temporal-gnn-project/assess_paper_results/data_verification/processed_data_investigation.json', 'w') as f:
            json.dump(investigation_report, f, indent=2)
            
        print(f"✅ Saved: processed_data_investigation.json")
        return investigation_report
        
    def run_complete_investigation(self):
        """Run complete processed data investigation"""
        print("🚀 STARTING PROCESSED DATA INVESTIGATION")
        print("="*60)
        
        # Analyze all files
        file_analysis = self.analyze_all_processed_files()
        
        # Create summary
        potential_matches = self.create_processed_data_summary(file_analysis)
        
        # Save results
        report = self.save_processed_investigation_results(file_analysis, potential_matches)
        
        # Final recommendation
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print("="*30)
        
        if potential_matches:
            best_match = potential_matches[0]
            filename, analysis = best_match
            
            print(f"✅ USE DATASET: {filename}")
            print(f"   This dataset matches paper claims for edges and users")
            if analysis['suspicious_users'] == 73:
                print(f"   🎯 PERFECT MATCH: Also has 73 suspicious users!")
                print(f"   Proceed with this dataset for all verification phases")
            else:
                print(f"   ⚠️ Suspicious users: {analysis['suspicious_users']} (paper claims 73)")
                print(f"   Investigate threshold criteria in processed data")
        else:
            print(f"❌ NO EXACT MATCHES FOUND")
            print(f"   Consider using combined dataset or investigate preprocessing")
            
        return len(potential_matches) > 0

def main():
    """Main processed data investigation"""
    investigator = ProcessedDataInvestigator()
    success = investigator.run_complete_investigation()
    
    if success:
        print(f"\n🎉 PROCESSED DATA INVESTIGATION SUCCESSFUL!")
        print(f"📋 Next: Use identified dataset for verification")
    else:
        print(f"\n⚠️ NO PERFECT MATCHES FOUND")
        print(f"📋 Need to investigate data preprocessing pipeline")
        
if __name__ == "__main__":
    main()
