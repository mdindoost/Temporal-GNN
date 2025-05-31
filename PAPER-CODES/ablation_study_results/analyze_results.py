#!/usr/bin/env python3
"""
Ablation Study Results Analysis
Run this AFTER you have collected real experimental data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_ablation_results(results_file='ablation_results_final.csv'):
    """Analyze ablation study results"""
    
    # Load your actual results
    df = pd.read_csv(results_file)
    
    # Group by configuration and compute statistics
    summary = df.groupby('config_name').agg({
        'early_detection': ['mean', 'std', 'count'],
        'cold_start': ['mean', 'std', 'count']
    }).round(3)
    
    print("ABLATION STUDY RESULTS SUMMARY:")
    print("="*50)
    print(summary)
    
    # Create comparison table for paper
    paper_table = []
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        
        early_mean = config_data['early_detection'].mean()
        early_std = config_data['early_detection'].std()
        cold_mean = config_data['cold_start'].mean()
        cold_std = config_data['cold_start'].std()
        
        # Get component weights
        alpha = config_data['alpha'].iloc[0]
        beta = config_data['beta'].iloc[0]
        gamma = config_data['gamma'].iloc[0]
        
        paper_table.append({
            'Configuration': config.replace('_', ' ').title(),
            'Weights': f"({alpha}, {beta}, {gamma})",
            'Early Detection': f"{early_mean:.3f} ± {early_std:.3f}",
            'Cold Start': f"{cold_mean:.3f} ± {cold_std:.3f}"
        })
    
    # Save paper-ready table
    df_paper = pd.DataFrame(paper_table)
    df_paper.to_csv('ablation_table_for_paper.csv', index=False)
    
    print("\n\nPAPER TABLE (LaTeX format):")
    print("="*50)
    
    # Generate LaTeX table
    latex_table = """
\begin{table}[htbp]
\centering
\caption{Ablation study on component weights showing interference effects.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Weights} & \textbf{Early Detection} & \textbf{Cold Start} \\
\midrule
"""
    
    for _, row in df_paper.iterrows():
        latex_table += f"{row['Configuration']} & {row['Weights']} & {row['Early Detection']} & {row['Cold Start']} \\\\
"
    
    latex_table += """\bottomrule
\end{tabular}
\end{table}
"""
    
    print(latex_table)
    
    # Save LaTeX table
    with open('ablation_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Early detection comparison
    plt.subplot(1, 2, 1)
    config_means = df.groupby('config_name')['early_detection'].mean()
    config_stds = df.groupby('config_name')['early_detection'].std()
    
    plt.bar(range(len(config_means)), config_means.values, 
            yerr=config_stds.values, capsize=5, alpha=0.7)
    plt.xticks(range(len(config_means)), config_means.index, rotation=45)
    plt.ylabel('Early Detection Performance')
    plt.title('Component Configuration Comparison')
    
    # Cold start comparison  
    plt.subplot(1, 2, 2)
    config_means = df.groupby('config_name')['cold_start'].mean()
    config_stds = df.groupby('config_name')['cold_start'].std()
    
    plt.bar(range(len(config_means)), config_means.values,
            yerr=config_stds.values, capsize=5, alpha=0.7)
    plt.xticks(range(len(config_means)), config_means.index, rotation=45)
    plt.ylabel('Cold Start Performance')
    plt.title('Component Configuration Comparison')
    
    plt.tight_layout()
    plt.savefig('ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_paper

if __name__ == "__main__":
    # Run analysis after you have collected data
    results_table = analyze_ablation_results()
    print("\nAnalysis complete! Check ablation_table.tex for paper content.")
