"""
Evaluation utilities for temporal GNN anomaly detection project
Contains metrics, visualization, and analysis functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class AnomalyEvaluator:
    """Comprehensive evaluation for anomaly detection"""
    
    def __init__(self, save_dir: str = "results/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def compute_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, 
                       threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute comprehensive anomaly detection metrics
        
        Args:
            y_true: True binary labels (1 for anomaly, 0 for normal)
            y_scores: Anomaly scores (higher = more anomalous)
            threshold: Decision threshold (if None, uses best F1)
            
        Returns:
            Dictionary of metrics
        """
        if len(np.unique(y_true)) < 2:
            return {
                "auc": 0.5, "ap": np.mean(y_true), "precision": 0.0, 
                "recall": 0.0, "f1": 0.0, "threshold": 0.5
            }
        
        # ROC AUC and Average Precision
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        # Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find best threshold based on F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        
        if threshold is None:
            best_threshold = thresholds[best_f1_idx] if len(thresholds) > best_f1_idx else 0.5
        else:
            best_threshold = threshold
        
        # Compute metrics at best threshold
        y_pred = (y_scores >= best_threshold).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_pred)) < 2:
            precision = recall = f1 = 0.0
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "auc": auc,
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": best_threshold
        }
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      title: str = "ROC Curve", save_name: str = "roc_curve.png"):
        """Plot ROC curve"""
        if len(np.unique(y_true)) < 2:
            print("Cannot plot ROC curve: only one class present")
            return
            
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                     title: str = "Precision-Recall Curve", save_name: str = "pr_curve.png"):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) < 2:
            print("Cannot plot PR curve: only one class present")
            return
            
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
        plt.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, 
                   label=f'Random Classifier (AP = {np.mean(y_true):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_distribution(self, scores_normal: np.ndarray, scores_anomaly: np.ndarray,
                               title: str = "Anomaly Score Distribution", 
                               save_name: str = "score_distribution.png"):
        """Plot distribution of anomaly scores for normal vs anomalous samples"""
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        plt.hist(scores_normal, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
        plt.hist(scores_anomaly, bins=50, alpha=0.7, label='Anomaly', density=True, color='red')
        
        # Add vertical lines for means
        plt.axvline(np.mean(scores_normal), color='blue', linestyle='--', 
                   label=f'Normal Mean = {np.mean(scores_normal):.3f}')
        plt.axvline(np.mean(scores_anomaly), color='red', linestyle='--',
                   label=f'Anomaly Mean = {np.mean(scores_anomaly):.3f}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_scores(self, timestamps: List[int], scores: List[float], 
                           anomaly_timestamps: List[int] = None,
                           title: str = "Anomaly Scores Over Time",
                           save_name: str = "temporal_scores.png"):
        """Plot anomaly scores over time"""
        plt.figure(figsize=(12, 6))
        
        # Plot scores
        plt.plot(timestamps, scores, 'b-', linewidth=2, label='Anomaly Score', marker='o', markersize=4)
        
        # Mark known anomalies
        if anomaly_timestamps:
            for t in anomaly_timestamps:
                if t in timestamps:
                    idx = timestamps.index(t)
                    plt.axvline(x=t, color='red', linestyle=':', alpha=0.7, linewidth=2)
                    plt.plot(t, scores[idx], 'ro', markersize=10, 
                            label='Known Anomaly' if t == anomaly_timestamps[0] else "")
        
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results: Dict, save_name: str = "evaluation_report.txt"):
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("ANOMALY DETECTION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 30)
        for metric, value in results.items():
            if isinstance(value, float):
                report.append(f"{metric.upper():>12}: {value:.4f}")
        report.append("")
        
        # Performance interpretation
        report.append("PERFORMANCE INTERPRETATION:")
        report.append("-" * 30)
        
        auc = results.get('auc', 0)
        if auc >= 0.9:
            report.append("• Excellent detection performance (AUC ≥ 0.9)")
        elif auc >= 0.8:
            report.append("• Good detection performance (0.8 ≤ AUC < 0.9)")
        elif auc >= 0.7:
            report.append("• Fair detection performance (0.7 ≤ AUC < 0.8)")
        else:
            report.append("• Poor detection performance (AUC < 0.7)")
        
        ap = results.get('ap', 0)
        if ap >= 0.8:
            report.append("• High precision-recall performance (AP ≥ 0.8)")
        elif ap >= 0.6:
            report.append("• Moderate precision-recall performance (0.6 ≤ AP < 0.8)")
        else:
            report.append("• Low precision-recall performance (AP < 0.6)")
        
        f1 = results.get('f1', 0)
        if f1 >= 0.8:
            report.append("• High balanced performance (F1 ≥ 0.8)")
        elif f1 >= 0.6:
            report.append("• Moderate balanced performance (0.6 ≤ F1 < 0.8)")
        else:
            report.append("• Low balanced performance (F1 < 0.6)")
        
        report.append("")
        
        # Save report
        with open(os.path.join(self.save_dir, save_name), 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        
        return report


class TemporalAnalyzer:
    """Analyze temporal patterns in anomaly detection"""
    
    def __init__(self, save_dir: str = "results/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def analyze_temporal_patterns(self, temporal_scores: Dict[int, np.ndarray], 
                                 anomaly_timestamps: List[int]) -> Dict[str, float]:
        """Analyze temporal patterns in anomaly scores"""
        
        # Extract timestamp-level statistics
        timestamps = sorted(temporal_scores.keys())
        mean_scores = [np.mean(temporal_scores[t]) for t in timestamps]
        max_scores = [np.max(temporal_scores[t]) for t in timestamps]
        std_scores = [np.std(temporal_scores[t]) for t in timestamps]
        
        # Separate normal and anomaly periods
        normal_mean_scores = [mean_scores[i] for i, t in enumerate(timestamps) if t not in anomaly_timestamps]
        anomaly_mean_scores = [mean_scores[i] for i, t in enumerate(timestamps) if t in anomaly_timestamps]
        
        normal_max_scores = [max_scores[i] for i, t in enumerate(timestamps) if t not in anomaly_timestamps]
        anomaly_max_scores = [max_scores[i] for i, t in enumerate(timestamps) if t in anomaly_timestamps]
        
        # Compute temporal statistics
        results = {
            'temporal_mean_separation': np.mean(anomaly_mean_scores) - np.mean(normal_mean_scores) if anomaly_mean_scores and normal_mean_scores else 0,
            'temporal_max_separation': np.mean(anomaly_max_scores) - np.mean(normal_max_scores) if anomaly_max_scores and normal_max_scores else 0,
            'normal_score_stability': np.std(normal_mean_scores) if normal_mean_scores else 0,
            'anomaly_score_variance': np.std(anomaly_mean_scores) if anomaly_mean_scores else 0,
            'detection_rate': len([t for t in anomaly_timestamps if mean_scores[timestamps.index(t)] > np.mean(normal_mean_scores) if t in timestamps]) / len(anomaly_timestamps) if anomaly_timestamps else 0
        }
        
        return results
    
    def plot_temporal_heatmap(self, temporal_scores: Dict[int, np.ndarray],
                             anomaly_timestamps: List[int] = None,
                             save_name: str = "temporal_heatmap.png"):
        """Create heatmap of node anomaly scores over time"""
        
        # Prepare data for heatmap
        timestamps = sorted(temporal_scores.keys())
        max_nodes = max(len(temporal_scores[t]) for t in timestamps)
        
        # Create matrix (timestamps x nodes)
        score_matrix = np.zeros((len(timestamps), max_nodes))
        
        for i, t in enumerate(timestamps):
            scores = temporal_scores[t]
            score_matrix[i, :len(scores)] = scores
            # Fill missing nodes with NaN
            if len(scores) < max_nodes:
                score_matrix[i, len(scores):] = np.nan
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Mask NaN values
        masked_matrix = np.ma.masked_invalid(score_matrix)
        
        im = plt.imshow(masked_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Anomaly Score')
        
        # Mark anomaly timestamps
        if anomaly_timestamps:
            for t in anomaly_timestamps:
                if t in timestamps:
                    idx = timestamps.index(t)
                    plt.axhline(y=idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        plt.xlabel('Node Index')
        plt.ylabel('Timestamp')
        plt.title('Node Anomaly Scores Over Time')
        
        # Set y-tick labels to actual timestamps
        y_ticks = range(0, len(timestamps), max(1, len(timestamps)//10))
        plt.yticks(y_ticks, [timestamps[i] for i in y_ticks])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()


def create_experiment_summary(results_df: pd.DataFrame, metrics: Dict[str, float],
                             config: Dict, save_dir: str = "results/") -> None:
    """Create a comprehensive experiment summary"""
    
    summary = []
    summary.append("=" * 80)
    summary.append("STATIC GNN BASELINE EXPERIMENT SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Configuration
    summary.append("EXPERIMENT CONFIGURATION:")
    summary.append("-" * 40)
    summary.append(f"Model Type: {config.get('model_type', 'DOMINANT')}")
    summary.append(f"Feature Dimension: {config.get('feature_dim', 'N/A')}")
    summary.append(f"Hidden Dimension: {config.get('hidden_dim', 'N/A')}")
    summary.append(f"Embedding Dimension: {config.get('embedding_dim', 'N/A')}")
    summary.append(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
    summary.append(f"Training Epochs: {config.get('epochs', 'N/A')}")
    summary.append("")
    
    # Data statistics
    summary.append("DATA STATISTICS:")
    summary.append("-" * 40)
    summary.append(f"Total Timestamps: {len(results_df)}")
    summary.append(f"Anomaly Timestamps: {results_df['is_anomaly'].sum()}")
    summary.append(f"Normal Timestamps: {(~results_df['is_anomaly']).sum()}")
    summary.append(f"Avg Nodes per Graph: {results_df['num_nodes'].mean():.1f}")
    summary.append(f"Avg Edges per Graph: {results_df['num_edges'].mean():.1f}")
    summary.append("")
    
    # Performance metrics
    summary.append("PERFORMANCE METRICS:")
    summary.append("-" * 40)
    for metric, value in metrics.items():
        summary.append(f"{metric.upper():>12}: {value:.4f}")
    summary.append("")
    
    # Top anomalous timestamps
    summary.append("TOP ANOMALOUS TIMESTAMPS:")
    summary.append("-" * 40)
    top_anomalies = results_df.nlargest(5, 'max_score')[['timestamp', 'is_anomaly', 'max_score', 'mean_score']]
    for _, row in top_anomalies.iterrows():
        status = "✓ Known" if row['is_anomaly'] else "✗ False"
        summary.append(f"T={row['timestamp']:2d}: {status} | Max={row['max_score']:.4f}, Mean={row['mean_score']:.4f}")
    
    summary.append("")
    summary.append("=" * 80)
    
    # Save summary
    with open(os.path.join(save_dir, "experiment_summary.txt"), 'w') as f:
        f.write('\n'.join(summary))
    
    # Print summary
    print('\n'.join(summary))


# Example usage and testing functions
def test_evaluator():
    """Test the evaluation utilities"""
    print("Testing AnomalyEvaluator...")
    
    # Generate synthetic test data
    np.random.seed(42)
    n_normal, n_anomaly = 100, 20
    
    # Normal samples: low scores
    normal_scores = np.random.beta(2, 5, n_normal)
    
    # Anomaly samples: high scores
    anomaly_scores = np.random.beta(5, 2, n_anomaly)
    
    # Combine
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    y_scores = np.concatenate([normal_scores, anomaly_scores])
    
    # Test evaluator
    evaluator = AnomalyEvaluator(save_dir="test_results/")
    
    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_scores)
    print("Metrics:", metrics)
    
    # Generate plots
    evaluator.plot_roc_curve(y_true, y_scores, title="Test ROC Curve")
    evaluator.plot_pr_curve(y_true, y_scores, title="Test PR Curve")
    evaluator.plot_score_distribution(normal_scores, anomaly_scores, title="Test Score Distribution")
    
    # Generate report
    evaluator.generate_report(metrics, "test_report.txt")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_evaluator()
