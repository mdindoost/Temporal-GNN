#!/usr/bin/env python3
"""
Temporal Anomaly Detector - Full Integration
Combines static baseline with temporal memory modules for enhanced anomaly detection

This script integrates:
1. Static DOMINANT baseline (for comparison)
2. Temporal memory modules (TGN/DyRep/JODIE inspired)
3. Unified training and evaluation pipeline
4. Comprehensive comparison and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import os
import time
from typing import Dict, List, Tuple, Optional

# Import our modules
from temporal_memory_module import TemporalAnomalyMemory, Time2Vec
from static_gnn_baseline import DOMINANTModel, StaticAnomalyDetector, evaluate_anomaly_detection

class TemporalAnomalyDetector:
    """
    Complete temporal anomaly detection system
    Combines static baseline with temporal memory for comprehensive anomaly detection
    """
    
    def __init__(self, num_nodes: int, node_feature_dim: int, 
                 hidden_dim: int = 64, embedding_dim: int = 32,
                 learning_rate: float = 0.01, device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Initialize temporal memory system
        self.temporal_memory = TemporalAnomalyMemory(
            num_nodes, node_feature_dim, hidden_dim, embedding_dim
        )
        
        # Move components to device
        self.temporal_memory.node_memory = self.temporal_memory.node_memory.to(self.device)
        self.temporal_memory.graph_memory = self.temporal_memory.graph_memory.to(self.device)
        self.temporal_memory.temporal_encoder = self.temporal_memory.temporal_encoder.to(self.device)
        self.temporal_memory.trajectory_predictor = self.temporal_memory.trajectory_predictor.to(self.device)
        
        # Initialize static baseline for comparison
        self.static_detector = StaticAnomalyDetector(
            node_feature_dim, hidden_dim, embedding_dim, learning_rate, device
        )
        
        # Training components
        self.optimizer = optim.Adam(self.get_temporal_parameters(), lr=learning_rate)
        self.temporal_criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'temporal_losses': [],
            'static_losses': [],
            'temporal_scores': [],
            'static_scores': []
        }
        
        print(f"Initialized Temporal Anomaly Detector on {self.device}")
        print(f"Temporal parameters: {sum(p.numel() for p in self.get_temporal_parameters()):,}")
        
    def get_temporal_parameters(self):
        """Get all trainable parameters from temporal components"""
        params = []
        params.extend(list(self.temporal_memory.temporal_encoder.parameters()))
        params.extend(list(self.temporal_memory.trajectory_predictor.parameters()))
        params.extend(list(self.temporal_memory.node_memory.parameters()))
        params.extend(list(self.temporal_memory.graph_memory.parameters()))
        return params
    
    def dict_to_pyg_data(self, graph_dict: Dict, feature_dim: int = 16) -> Data:
        """Convert graph dictionary to PyTorch Geometric Data object"""
        num_nodes = graph_dict['num_nodes']
        edges = graph_dict['edges']
        
        if num_nodes == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            x = torch.empty((0, feature_dim), dtype=torch.float)
            return Data(x=x, edge_index=edge_index)
        
        # Use provided node features if available, otherwise create structural features
        if 'node_features' in graph_dict and graph_dict['node_features'] is not None:
            features = np.array(graph_dict['node_features'])
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            
            # Pad or truncate to desired feature dimension
            if features.shape[1] < feature_dim:
                padding = np.zeros((features.shape[0], feature_dim - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            elif features.shape[1] > feature_dim:
                features = features[:, :feature_dim]
        else:
            # Create basic random features (placeholder)
            features = np.random.randn(num_nodes, feature_dim) * 0.1
        
        # Create edge index
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Make undirected by adding reverse edges
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create node features tensor
        x = torch.tensor(features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)
    
    def temporal_training_step(self, graph_data: Data, timestamp: float, 
                              is_normal: bool = True) -> float:
        """
        Single training step for temporal components
        
        Args:
            graph_data: PyTorch Geometric graph data
            timestamp: Current timestamp
            is_normal: Whether this is normal behavior
            
        Returns:
            training_loss: Loss value for this step
        """
        self.optimizer.zero_grad()
        
        # Move data to device
        graph_data = graph_data.to(self.device)
        
        # Process graph through temporal memory
        results = self.temporal_memory.process_graph(
            graph_data.x, graph_data.edge_index, timestamp, is_normal
        )
        
        # Compute temporal consistency loss
        # For normal graphs, we want low anomaly scores
        # For anomalous graphs, we want high anomaly scores (during validation only)
        
        if is_normal:
            # Normal graphs should have low anomaly scores
            memory_loss = torch.mean(results['node_memory_scores']) + results['graph_memory_score']
            target_score = torch.tensor(0.1, device=self.device)  # Low target for normal
            consistency_loss = F.mse_loss(memory_loss, target_score)
        else:
            # Anomalous graphs should have high scores (but we don't train on these)
            consistency_loss = torch.tensor(0.0, device=self.device)
        
        # Embedding regularization
        embedding_reg = 0.001 * torch.norm(results['node_embeddings'], p=2)
        
        # Total loss
        total_loss = consistency_loss + embedding_reg
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_temporal_parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train_temporal_model(self, temporal_data: List[Dict], epochs: int = 50, 
                           validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the temporal anomaly detection model
        
        Args:
            temporal_data: List of graph dictionaries with temporal information
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            
        Returns:
            training_history: Dictionary of training metrics
        """
        print(f"Training temporal model for {epochs} epochs...")
        
        # Separate normal and anomalous data
        normal_data = [g for g in temporal_data if not g['is_anomaly']]
        anomalous_data = [g for g in temporal_data if g['is_anomaly']]
        
        # Split normal data for training/validation
        split_idx = int(len(normal_data) * (1 - validation_split))
        train_normal = normal_data[:split_idx]
        val_normal = normal_data[split_idx:]
        
        print(f"Training on {len(train_normal)} normal graphs")
        print(f"Validation on {len(val_normal)} normal + {len(anomalous_data)} anomalous graphs")
        
        training_losses = []
        validation_scores = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.temporal_memory.is_training_phase = True
            epoch_losses = []
            
            # Shuffle training data
            np.random.shuffle(train_normal)
            
            for graph_dict in train_normal:
                # Convert to PyG format
                graph_data = self.dict_to_pyg_data(graph_dict, self.node_feature_dim)
                
                # Skip empty graphs
                if graph_data.x.shape[0] == 0:
                    continue
                
                # Training step
                loss = self.temporal_training_step(
                    graph_data, float(graph_dict['timestamp']), is_normal=True
                )
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            training_losses.append(avg_loss)
            
            # Validation phase (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.temporal_memory.is_training_phase = False
                val_scores = self.evaluate_on_validation(val_normal + anomalous_data)
                validation_scores.append(val_scores['auc'])
                
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss = {avg_loss:.6f}, "
                      f"Val AUC = {val_scores['auc']:.4f}, "
                      f"Time = {epoch_time:.2f}s")
            else:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss = {avg_loss:.6f}, "
                      f"Time = {epoch_time:.2f}s")
        
        # Store training history
        self.training_history['temporal_losses'] = training_losses
        self.training_history['validation_aucs'] = validation_scores
        
        print("Temporal training completed!")
        return self.training_history
    
    def evaluate_on_validation(self, val_data: List[Dict]) -> Dict[str, float]:
        """Evaluate temporal model on validation data"""
        scores = []
        labels = []
        
        with torch.no_grad():
            for graph_dict in val_data:
                # Convert to PyG format
                graph_data = self.dict_to_pyg_data(graph_dict, self.node_feature_dim)
                
                if graph_data.x.shape[0] == 0:
                    continue
                
                # Move to device
                graph_data = graph_data.to(self.device)
                
                # Process graph
                results = self.temporal_memory.process_graph(
                    graph_data.x, graph_data.edge_index, 
                    float(graph_dict['timestamp']), is_normal=False
                )
                
                # Compute unified anomaly score
                score = self.temporal_memory.compute_unified_anomaly_score(results)
                
                scores.append(score.item())
                labels.append(int(graph_dict['is_anomaly']))
        
        # Compute metrics
        scores = np.array(scores)
        labels = np.array(labels)
        
        if len(np.unique(labels)) > 1:
            metrics = evaluate_anomaly_detection(scores, labels)
        else:
            metrics = {'auc': 0.5, 'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        return metrics
    
    def compare_with_static_baseline(self, temporal_data: List[Dict]) -> Dict[str, Dict]:
        """
        Compare temporal approach with static baseline on the same data
        
        Args:
            temporal_data: List of temporal graph data
            
        Returns:
            comparison_results: Performance comparison
        """
        print("\nComparing Temporal vs Static Approaches...")
        
        # Prepare data for static baseline
        static_data = []
        for graph_dict in temporal_data:
            pyg_data = self.dict_to_pyg_data(graph_dict, self.node_feature_dim)
            if pyg_data.x.shape[0] > 0:
                static_data.append(pyg_data)
        
        # Train static baseline on normal data
        normal_static = [static_data[i] for i, g in enumerate(temporal_data) 
                        if not g['is_anomaly'] and static_data[i] is not None]
        
        if len(normal_static) > 0:
            static_loader = DataLoader(normal_static[:30], batch_size=1, shuffle=True)
            print("Training static baseline...")
            self.static_detector.fit(static_loader, epochs=50)
        
        # Evaluate both approaches
        temporal_scores = []
        static_scores = []
        labels = []
        
        print("\nEvaluating both approaches...")
        
        with torch.no_grad():
            for i, graph_dict in enumerate(temporal_data):
                if i >= len(static_data) or static_data[i] is None:
                    continue
                
                pyg_data = static_data[i].to(self.device)
                
                # Temporal approach
                results = self.temporal_memory.process_graph(
                    pyg_data.x, pyg_data.edge_index, 
                    float(graph_dict['timestamp']), is_normal=False
                )
                temporal_score = self.temporal_memory.compute_unified_anomaly_score(results)
                temporal_scores.append(temporal_score.item())
                
                # Static approach
                static_node_scores, _ = self.static_detector.compute_anomaly_scores(
                    pyg_data.x, pyg_data.edge_index
                )
                static_score = np.mean(static_node_scores)
                static_scores.append(static_score)
                
                # Label
                labels.append(int(graph_dict['is_anomaly']))
        
        # Compute metrics for both
        temporal_scores = np.array(temporal_scores)
        static_scores = np.array(static_scores)
        labels = np.array(labels)
        
        temporal_metrics = evaluate_anomaly_detection(temporal_scores, labels)
        static_metrics = evaluate_anomaly_detection(static_scores, labels)
        
        comparison = {
            'temporal': temporal_metrics,
            'static': static_metrics,
            'improvement': {
                'auc': temporal_metrics['auc'] - static_metrics['auc'],
                'ap': temporal_metrics['ap'] - static_metrics['ap'],
                'f1': temporal_metrics['f1'] - static_metrics['f1']
            }
        }
        
        return comparison
    
    def visualize_temporal_results(self, temporal_data: List[Dict], 
                                  comparison_results: Dict, save_path: str = 'results/'):
        """
        Create comprehensive visualizations of temporal anomaly detection results
        
        Args:
            temporal_data: Temporal graph data
            comparison_results: Results from temporal vs static comparison
            save_path: Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Plot 1: Training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['temporal_losses'])
        plt.title('Temporal Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if 'validation_aucs' in self.training_history:
            epochs_val = range(4, len(self.training_history['temporal_losses']), 5)
            plt.plot(epochs_val, self.training_history['validation_aucs'])
            plt.title('Validation AUC During Training')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'temporal_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Performance comparison
        metrics = ['auc', 'ap', 'precision', 'recall', 'f1']
        temporal_values = [comparison_results['temporal'][m] for m in metrics]
        static_values = [comparison_results['static'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, static_values, width, label='Static Baseline', alpha=0.8)
        plt.bar(x + width/2, temporal_values, width, label='Temporal Model', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Temporal vs Static Anomaly Detection Performance')
        plt.xticks(x, [m.upper() for m in metrics])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, (t_val, s_val) in enumerate(zip(temporal_values, static_values)):
            improvement = t_val - s_val
            plt.annotate(f'+{improvement:.3f}', 
                        xy=(i + width/2, t_val), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=8, color='green' if improvement > 0 else 'red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'temporal_vs_static_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Temporal anomaly scores over time
        timestamps = []
        temporal_scores = []
        is_anomaly_list = []
        
        with torch.no_grad():
            for graph_dict in temporal_data:
                pyg_data = self.dict_to_pyg_data(graph_dict, self.node_feature_dim)
                if pyg_data.x.shape[0] == 0:
                    continue
                
                pyg_data = pyg_data.to(self.device)
                results = self.temporal_memory.process_graph(
                    pyg_data.x, pyg_data.edge_index, 
                    float(graph_dict['timestamp']), is_normal=False
                )
                score = self.temporal_memory.compute_unified_anomaly_score(results)
                
                timestamps.append(graph_dict['timestamp'])
                temporal_scores.append(score.item())
                is_anomaly_list.append(graph_dict['is_anomaly'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, temporal_scores, 'b-', linewidth=2, marker='o', markersize=4, 
                label='Temporal Anomaly Score')
        
        # Mark known anomalies
        anomaly_timestamps = [t for t, is_anom in zip(timestamps, is_anomaly_list) if is_anom]
        anomaly_scores = [s for s, is_anom in zip(temporal_scores, is_anomaly_list) if is_anom]
        
        for t, s in zip(anomaly_timestamps, anomaly_scores):
            plt.axvline(x=t, color='red', linestyle=':', alpha=0.7, linewidth=2)
            plt.plot(t, s, 'ro', markersize=10, 
                    label='Known Anomaly' if t == anomaly_timestamps[0] else "")
        
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score')
        plt.title('Temporal Anomaly Detection Results Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'temporal_anomaly_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_path}")

def load_synthetic_temporal_data(data_path: str = 'data/synthetic/') -> Tuple[List[Dict], List[int]]:
    """Load synthetic temporal graph data"""
    print("Loading synthetic temporal graph data...")
    
    pkl_file = os.path.join(data_path, 'temporal_graph_with_anomalies.pkl')
    csv_file = os.path.join(data_path, 'temporal_graph_summary.csv')
    
    with open(pkl_file, 'rb') as f:
        temporal_data = pickle.load(f)
    
    summary_df = pd.read_csv(csv_file)
    known_anomalies = summary_df[summary_df['is_anomaly']]['timestamp'].tolist()
    
    print(f"Loaded {len(temporal_data)} timestamps")
    print(f"Known anomalies at: {known_anomalies}")
    
    return temporal_data, known_anomalies

def main():
    """Main execution function for temporal anomaly detection"""
    print("="*60)
    print("TEMPORAL GRAPH NEURAL NETWORK ANOMALY DETECTION")
    print("="*60)
    
    # Configuration
    config = {
        'data_path': 'data/synthetic/',
        'node_feature_dim': 16,
        'hidden_dim': 64,
        'embedding_dim': 32,
        'learning_rate': 0.01,
        'epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {config}")
    
    try:
        # Load temporal data
        temporal_data, known_anomalies = load_synthetic_temporal_data(config['data_path'])
        
        # Initialize temporal detector
        detector = TemporalAnomalyDetector(
            num_nodes=100,  # Based on synthetic data
            node_feature_dim=config['node_feature_dim'],
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            learning_rate=config['learning_rate'],
            device=config['device']
        )
        
        # Train temporal model
        print("\n" + "="*40)
        print("TEMPORAL MODEL TRAINING")
        print("="*40)
        
        training_history = detector.train_temporal_model(
            temporal_data, epochs=config['epochs']
        )
        
        # Compare with static baseline
        print("\n" + "="*40)
        print("TEMPORAL VS STATIC COMPARISON")
        print("="*40)
        
        comparison_results = detector.compare_with_static_baseline(temporal_data)
        
        # Print results
        print("\nFINAL RESULTS:")
        print("-" * 30)
        print("Static Baseline:")
        for metric, value in comparison_results['static'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nTemporal Model:")
        for metric, value in comparison_results['temporal'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nImprovement:")
        for metric, value in comparison_results['improvement'].items():
            print(f"  {metric.upper()}: {value:+.4f}")
        
        # Create visualizations
        print("\n" + "="*40)
        print("GENERATING VISUALIZATIONS")
        print("="*40)
        
        detector.visualize_temporal_results(temporal_data, comparison_results)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Save comparison results
        with open('results/temporal_comparison_results.txt', 'w') as f:
            f.write("TEMPORAL GNN ANOMALY DETECTION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Static Baseline Results:\n")
            for metric, value in comparison_results['static'].items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Temporal Model Results:\n")
            for metric, value in comparison_results['temporal'].items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Improvements:\n")
            for metric, value in comparison_results['improvement'].items():
                f.write(f"  {metric.upper()}: {value:+.4f}\n")
        
        print("✅ Temporal anomaly detection completed successfully!")
        print("📊 Results saved to 'results/' directory")
        
        # Calculate success metrics
        auc_improvement = comparison_results['improvement']['auc']
        temporal_auc = comparison_results['temporal']['auc']
        
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        print(f"   • Temporal AUC: {temporal_auc:.4f}")
        print(f"   • AUC Improvement: {auc_improvement:+.4f}")
        print(f"   • Target AUC (>0.80): {'✅ Achieved' if temporal_auc > 0.80 else '❌ Not yet'}")
        print(f"   • Improvement over static: {'✅ Success' if auc_improvement > 0 else '❌ Need tuning'}")
        
        return detector, comparison_results
        
    except Exception as e:
        print(f"❌ Error in temporal anomaly detection: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    detector, results = main()
