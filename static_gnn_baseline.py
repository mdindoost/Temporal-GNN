#!/usr/bin/env python3
"""
Corrected Static GNN Autoencoder Baseline (DOMINANT-style)
Handles the specific format of synthetic temporal graph data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import os
import time
from typing import Tuple, List, Dict, Optional

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GCNEncoder(nn.Module):
    """Graph Convolutional Network Encoder for node embeddings"""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.2):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, x, edge_index):
        # First GCN layer
        h1 = self.conv1(x, edge_index)
        h1 = self.batch_norm1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        
        # Second GCN layer
        h2 = self.conv2(h1, edge_index)
        h2 = self.batch_norm2(h2)
        embeddings = F.relu(h2)
        
        return embeddings

class InnerProductDecoder(nn.Module):
    """Inner product decoder for edge reconstruction"""
    
    def __init__(self, dropout: float = 0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings, edge_index):
        # Get embeddings for source and target nodes
        row, col = edge_index
        source_embeddings = embeddings[row]
        target_embeddings = embeddings[col]
        
        # Apply dropout
        source_embeddings = self.dropout(source_embeddings)
        target_embeddings = self.dropout(target_embeddings)
        
        # Inner product for edge reconstruction
        edge_probs = torch.sum(source_embeddings * target_embeddings, dim=1)
        edge_probs = torch.sigmoid(edge_probs)
        
        return edge_probs

class DOMINANTModel(nn.Module):
    """DOMINANT-style Graph Autoencoder for Anomaly Detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32, 
                 dropout: float = 0.2):
        super(DOMINANTModel, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim, dropout)
        self.decoder = InnerProductDecoder(dropout)
        self.embedding_dim = embedding_dim
        
    def forward(self, x, edge_index):
        # Encode nodes to embeddings
        embeddings = self.encoder(x, edge_index)
        
        # Decode edges from embeddings
        edge_probs = self.decoder(embeddings, edge_index)
        
        return embeddings, edge_probs

class StaticAnomalyDetector:
    """Static Graph Anomaly Detector using DOMINANT approach"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32,
                 learning_rate: float = 0.01, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DOMINANTModel(input_dim, hidden_dim, embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        
        print(f"Initialized DOMINANT model on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def compute_anomaly_scores(self, x, edge_index):
        """Compute anomaly scores based on reconstruction error"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            
            # Get embeddings and reconstructed edge probabilities
            embeddings, edge_probs = self.model(x, edge_index)
            
            # Compute reconstruction errors for edges
            edge_errors = 1.0 - edge_probs
            
            # Compute node-level anomaly scores
            node_scores = torch.zeros(x.shape[0], device=self.device)
            
            # Aggregate edge errors to node level
            row, col = edge_index
            for i in range(x.shape[0]):
                # Find edges connected to node i
                mask = (row == i) | (col == i)
                if mask.sum() > 0:
                    node_scores[i] = edge_errors[mask].mean()
                else:
                    # Isolated nodes get high anomaly score
                    node_scores[i] = 1.0
            
            return node_scores.cpu().numpy(), embeddings.cpu().numpy()
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings, edge_probs = self.model(batch.x, batch.edge_index)
            
            # Create target labels (all edges should be reconstructed as 1)
            targets = torch.ones_like(edge_probs)
            
            # Compute loss
            loss = self.criterion(edge_probs, targets)
            
            # Add regularization on embeddings
            reg_loss = 0.001 * torch.norm(embeddings, p=2)
            total_loss_batch = loss + reg_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def fit(self, train_loader, epochs: int = 100, patience: int = 10):
        """Train the model"""
        print(f"Training DOMINANT model for {epochs} epochs...")
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            epoch_time = time.time() - start_time
            
            # Early stopping
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_dominant_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss = {train_loss:.6f}, "
                      f"Time = {epoch_time:.2f}s")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_dominant_model.pth'))
        print("Training completed. Best model loaded.")
        
        return train_losses

def load_synthetic_data(data_path: str = 'data/synthetic/'):
    """Load synthetic temporal graph data from the specific format"""
    print("Loading synthetic temporal graph data...")
    
    # Load the temporal graph
    pkl_file = os.path.join(data_path, 'temporal_graph_with_anomalies.pkl')
    csv_file = os.path.join(data_path, 'temporal_graph_summary.csv')
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        temporal_data = pickle.load(f)
    
    print(f"Loaded temporal graph with {len(temporal_data)} timestamps")
    
    # Load summary
    summary_df = pd.read_csv(csv_file)
    print(f"Summary file columns: {list(summary_df.columns)}")
    
    # Get known anomalies
    known_anomalies = summary_df[summary_df['is_anomaly']]['timestamp'].tolist()
    print(f"Known anomalies at timestamps: {known_anomalies}")
    
    return temporal_data, summary_df, known_anomalies

def dict_to_networkx(graph_dict: Dict) -> nx.Graph:
    """Convert graph dictionary to NetworkX graph"""
    G = nx.Graph()
    
    # Add nodes
    num_nodes = graph_dict['num_nodes']
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    edges = graph_dict['edges']
    if len(edges) > 0:
        G.add_edges_from(edges)
    
    return G

def dict_to_pyg_data(graph_dict: Dict, feature_dim: int = 16) -> Data:
    """Convert graph dictionary directly to PyTorch Geometric Data object"""
    
    num_nodes = graph_dict['num_nodes']
    edges = graph_dict['edges']
    
    # Handle empty graphs
    if num_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.empty((0, feature_dim), dtype=torch.float)
        return Data(x=x, edge_index=edge_index)
    
    # Use provided node features if available, otherwise create structural features
    if 'node_features' in graph_dict and graph_dict['node_features'] is not None:
        features = np.array(graph_dict['node_features'])
        if len(features.shape) == 1:
            # If 1D, expand to 2D
            features = features.reshape(-1, 1)
        
        # Pad or truncate to desired feature dimension
        if features.shape[1] < feature_dim:
            # Pad with zeros
            padding = np.zeros((features.shape[0], feature_dim - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > feature_dim:
            # Truncate
            features = features[:, :feature_dim]
    else:
        # Create basic structural features
        G = dict_to_networkx(graph_dict)
        features = create_structural_features(G, feature_dim)
    
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

def create_structural_features(G: nx.Graph, feature_dim: int = 16) -> np.ndarray:
    """Create structural features from NetworkX graph"""
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0:
        return np.empty((0, feature_dim))
    
    features = np.zeros((num_nodes, feature_dim))
    
    # Basic structural features
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    
    # Node centrality measures
    try:
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
    except:
        # For disconnected graphs
        betweenness = {node: 0.0 for node in G.nodes()}
        closeness = {node: 0.0 for node in G.nodes()}
        pagerank = {node: 1.0/num_nodes for node in G.nodes()}
    
    node_list = list(G.nodes())
    
    for i, node in enumerate(node_list):
        max_degree = max(degrees.values()) if degrees else 1
        features[i, 0] = degrees.get(node, 0) / max_degree
        features[i, 1] = clustering.get(node, 0)
        features[i, 2] = betweenness.get(node, 0)
        features[i, 3] = closeness.get(node, 0)
        features[i, 4] = pagerank.get(node, 0)
        
        # Add some consistent random features
        np.random.seed(hash(str(node)) % 2**32)
        features[i, 5:] = np.random.randn(feature_dim - 5) * 0.1
    
    # Normalize features
    if num_nodes > 1:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    return features

def evaluate_anomaly_detection(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate anomaly detection performance"""
    if len(np.unique(labels)) < 2:
        return {"auc": 0.5, "ap": np.mean(labels), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # ROC AUC
    auc = roc_auc_score(labels, scores)
    
    # Average Precision
    ap = average_precision_score(labels, scores)
    
    # Precision-Recall at different thresholds
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    
    # Find best F1 score threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    
    return {
        "auc": auc,
        "ap": ap,
        "precision": precisions[best_f1_idx],
        "recall": recalls[best_f1_idx],
        "f1": f1_scores[best_f1_idx],
        "threshold": thresholds[best_f1_idx] if len(thresholds) > best_f1_idx else 0.5
    }

def visualize_results(temporal_data, summary_df, all_scores, known_anomalies, save_path: str = 'results/'):
    """Visualize anomaly detection results"""
    os.makedirs(save_path, exist_ok=True)
    
    # Plot 1: Anomaly scores over time
    timestamps = list(all_scores.keys())
    mean_scores = [np.mean(all_scores[t]) if len(all_scores[t]) > 0 else 0 for t in timestamps]
    max_scores = [np.max(all_scores[t]) if len(all_scores[t]) > 0 else 0 for t in timestamps]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, mean_scores, 'b-', label='Mean Anomaly Score', linewidth=2, marker='o', markersize=4)
    plt.plot(timestamps, max_scores, 'r--', label='Max Anomaly Score', linewidth=2, marker='s', markersize=4)
    
    # Mark known anomalies
    for t in known_anomalies:
        if t in timestamps:
            idx = timestamps.index(t)
            plt.axvline(x=t, color='red', linestyle=':', alpha=0.7, linewidth=2)
            plt.plot(t, mean_scores[idx], 'ro', markersize=10, 
                    label='Known Anomaly' if t == known_anomalies[0] else "")
    
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    plt.title('Static GNN Anomaly Detection Results Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'anomaly_scores_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Score distribution at anomaly vs normal times
    normal_scores = []
    anomaly_scores = []
    
    for t in timestamps:
        if t in known_anomalies:
            anomaly_scores.extend(all_scores[t])
        else:
            normal_scores.extend(all_scores[t])
    
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal Times', density=True, color='blue')
        plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly Times', density=True, color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'score_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("DOMINANT-Style Static GNN Baseline for Anomaly Detection")
    print("="*60)
    
    # Configuration
    config = {
        'data_path': 'data/synthetic/',
        'feature_dim': 16,
        'hidden_dim': 64,
        'embedding_dim': 32,
        'learning_rate': 0.01,
        'epochs': 50,  # Reduced for faster testing
        'batch_size': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {config}")
    
    try:
        # Load data
        temporal_data, summary_df, known_anomalies = load_synthetic_data(config['data_path'])
        
        # Prepare training data (use first 40 timestamps for training)
        train_data = []
        test_data = []
        
        print("\nPreparing graph data...")
        for i, graph_dict in enumerate(temporal_data):
            timestamp = graph_dict['timestamp']
            
            if graph_dict['num_nodes'] == 0:
                print(f"Skipping empty graph at timestamp {timestamp}")
                continue
            
            # Convert to PyG format
            pyg_data = dict_to_pyg_data(graph_dict, config['feature_dim'])
            
            if i < 40:  # Training data
                train_data.append(pyg_data)
            else:  # Test data
                test_data.append((timestamp, graph_dict, pyg_data))
        
        print(f"Training graphs: {len(train_data)}")
        print(f"Test graphs: {len(test_data)}")
        
        if len(train_data) == 0:
            raise ValueError("No training data available!")
        
        # Create data loader
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        
        # Initialize model
        detector = StaticAnomalyDetector(
            input_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            learning_rate=config['learning_rate'],
            device=config['device']
        )
        
        # Train model
        print("\n" + "="*40)
        print("TRAINING PHASE")
        print("="*40)
        
        train_losses = detector.fit(train_loader, epochs=config['epochs'])
        
        # Test model on all timestamps
        print("\n" + "="*40)
        print("EVALUATION PHASE")
        print("="*40)
        
        all_scores = {}
        results_summary = []
        
        for graph_dict in temporal_data:
            timestamp = graph_dict['timestamp']
            
            if graph_dict['num_nodes'] == 0:
                continue
            
            # Convert to PyG format
            pyg_data = dict_to_pyg_data(graph_dict, config['feature_dim'])
            
            # Compute anomaly scores
            scores, embeddings = detector.compute_anomaly_scores(
                pyg_data.x, pyg_data.edge_index
            )
            all_scores[timestamp] = scores
            
            # Check if this is a known anomaly timestamp
            is_anomaly = graph_dict['is_anomaly']
            mean_score = np.mean(scores) if len(scores) > 0 else 0
            max_score = np.max(scores) if len(scores) > 0 else 0
            
            results_summary.append({
                'timestamp': timestamp,
                'is_anomaly': is_anomaly,
                'anomaly_type': graph_dict['anomaly_type'],
                'mean_score': mean_score,
                'max_score': max_score,
                'num_nodes': graph_dict['num_nodes'],
                'num_edges': graph_dict['num_edges']
            })
            
            anomaly_symbol = "🚨" if is_anomaly else "✅"
            print(f"T={timestamp:2d}: {anomaly_symbol} {graph_dict['anomaly_type']:12s} | "
                  f"Mean={mean_score:.4f}, Max={max_score:.4f} | "
                  f"Nodes={graph_dict['num_nodes']:3d}, Edges={graph_dict['num_edges']:3d}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_summary)
        
        # Evaluate overall performance
        labels = results_df['is_anomaly'].astype(int).values
        mean_scores = results_df['mean_score'].values
        max_scores = results_df['max_score'].values
        
        print("\n" + "="*40)
        print("PERFORMANCE RESULTS")
        print("="*40)
        
        mean_metrics = evaluate_anomaly_detection(mean_scores, labels)
        max_metrics = evaluate_anomaly_detection(max_scores, labels)
        
        print("Using Mean Node Scores:")
        for metric, value in mean_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nUsing Max Node Scores:")
        for metric, value in max_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/static_baseline_results.csv', index=False)
        
        # Save metrics
        with open('results/metrics_summary.txt', 'w') as f:
            f.write("STATIC GNN BASELINE RESULTS\n")
            f.write("="*40 + "\n\n")
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nMean Node Scores:\n")
            for metric, value in mean_metrics.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            f.write("\nMax Node Scores:\n")
            for metric, value in max_metrics.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
        
        # Visualize results
        visualize_results(temporal_data, summary_df, all_scores, known_anomalies)
        
        print(f"\nResults saved to 'results/' directory")
        print("✅ Baseline implementation completed successfully!")
        
        return detector, results_df, all_scores
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    detector, results_df, all_scores = main()
