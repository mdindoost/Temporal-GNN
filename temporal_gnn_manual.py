"""
Manual implementation of temporal GNN components
This replaces torch_geometric_temporal if it's not available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import List, Optional, Tuple

class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network
    Combines GCN with LSTM for temporal modeling
    """
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int, 
                 output_dim: int,
                 num_gnn_layers: int = 2,
                 num_lstm_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # GNN layers for spatial encoding
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = node_features if i == 0 else hidden_dim
            self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_sequence, edge_index_sequence, hidden_state=None):
        """
        Forward pass through temporal GNN
        
        Args:
            x_sequence: List of node features [x_t1, x_t2, ..., x_tn]
            edge_index_sequence: List of edge indices [edge_t1, edge_t2, ..., edge_tn]
            hidden_state: Initial LSTM hidden state
            
        Returns:
            output: Final node embeddings
            hidden_state: Final LSTM hidden state
        """
        batch_size = len(x_sequence)
        
        # Process each timestamp with GNN
        spatial_embeddings = []
        
        for t in range(len(x_sequence)):
            x_t = x_sequence[t]
            edge_index_t = edge_index_sequence[t]
            
            # Apply GNN layers
            h = x_t
            for gnn_layer in self.gnn_layers:
                h = F.relu(gnn_layer(h, edge_index_t))
                h = self.dropout(h)
            
            spatial_embeddings.append(h)
        
        # Stack temporal embeddings: [num_nodes, seq_len, hidden_dim]
        if len(spatial_embeddings) > 1:
            # For multiple timestamps
            temporal_input = torch.stack(spatial_embeddings, dim=1)
            
            # Apply LSTM across time dimension
            lstm_out, hidden_state = self.lstm(temporal_input, hidden_state)
            
            # Use final timestamp output
            final_embedding = lstm_out[:, -1, :]
        else:
            # Single timestamp - no temporal modeling needed
            final_embedding = spatial_embeddings[0]
        
        # Final output
        output = self.output_layer(final_embedding)
        
        return output, hidden_state

class GraphAutoEncoder(nn.Module):
    """
    Graph Autoencoder for anomaly detection
    Reconstructs node features and graph structure
    """
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            GCNConv(hidden_dim, embedding_dim)
        )
        
        # Node feature decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x, edge_index):
        """Encode node features"""
        h = F.relu(self.encoder[0](x, edge_index))
        h = self.encoder[2](h)  # Dropout
        z = self.encoder[3](h, edge_index)
        return z
    
    def decode_nodes(self, z):
        """Decode node features"""
        return self.node_decoder(z)
    
    def decode_edges(self, z):
        """Decode edge probabilities using inner product"""
        return torch.sigmoid(torch.mm(z, z.t()))
    
    def forward(self, x, edge_index):
        """Full forward pass"""
        z = self.encode(x, edge_index)
        x_recon = self.decode_nodes(z)
        adj_recon = self.decode_edges(z)
        return z, x_recon, adj_recon

class TemporalAnomalyDetector(nn.Module):
    """
    Complete temporal anomaly detection model
    Combines temporal modeling with anomaly scoring
    """
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int = 64,
                 embedding_dim: int = 32,
                 sequence_length: int = 10):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Temporal GNN encoder
        self.temporal_encoder = TemporalGCN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_gnn_layers=2,
            num_lstm_layers=1
        )
        
        # Reconstruction components
        self.node_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )
        
        # Anomaly score predictor
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_sequence, edge_index_sequence):
        """
        Forward pass for anomaly detection
        
        Returns:
            embeddings: Node embeddings
            node_recon: Reconstructed node features
            edge_recon: Reconstructed adjacency matrix
            anomaly_scores: Anomaly scores for each node
        """
        # Get temporal embeddings
        embeddings, _ = self.temporal_encoder(x_sequence, edge_index_sequence)
        
        # Reconstruct node features
        node_recon = self.node_decoder(embeddings)
        
        # Reconstruct edges (inner product)
        edge_recon = torch.sigmoid(torch.mm(embeddings, embeddings.t()))
        
        # Predict anomaly scores
        anomaly_scores = self.anomaly_predictor(embeddings)
        
        return embeddings, node_recon, edge_recon, anomaly_scores

class TemporalDataLoader:
    """
    Custom data loader for temporal graphs
    Handles sequence creation and batching
    """
    def __init__(self, 
                 temporal_graphs: List,
                 sequence_length: int = 10,
                 step_size: int = 1):
        self.temporal_graphs = temporal_graphs
        self.sequence_length = sequence_length
        self.step_size = step_size
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """Create overlapping sequences from temporal data"""
        sequences = []
        
        for i in range(0, len(self.temporal_graphs) - self.sequence_length + 1, self.step_size):
            sequence = self.temporal_graphs[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def compute_anomaly_loss(model_output, target_data, reconstruction_weight=1.0, anomaly_weight=1.0):
    """
    Compute combined loss for temporal anomaly detection
    
    Args:
        model_output: Tuple of (embeddings, node_recon, edge_recon, anomaly_scores)
        target_data: Tuple of (target_nodes, target_edges, anomaly_labels)
        reconstruction_weight: Weight for reconstruction loss
        anomaly_weight: Weight for anomaly detection loss
    """
    embeddings, node_recon, edge_recon, anomaly_scores = model_output
    target_nodes, target_edges, anomaly_labels = target_data
    
    # Node reconstruction loss
    node_loss = F.mse_loss(node_recon, target_nodes)
    
    # Edge reconstruction loss
    edge_loss = F.binary_cross_entropy(edge_recon, target_edges)
    
    # Anomaly detection loss (if labels available)
    if anomaly_labels is not None:
        anomaly_loss = F.binary_cross_entropy(anomaly_scores.squeeze(), anomaly_labels.float())
    else:
        anomaly_loss = torch.tensor(0.0)
    
    # Combined loss
    total_loss = (reconstruction_weight * (node_loss + edge_loss) + 
                  anomaly_weight * anomaly_loss)
    
    return {
        'total_loss': total_loss,
        'node_loss': node_loss,
        'edge_loss': edge_loss,
        'anomaly_loss': anomaly_loss
    }

# Utility functions
def create_adjacency_matrix(edge_index, num_nodes):
    """Create dense adjacency matrix from edge_index"""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def temporal_train_test_split(temporal_data, train_ratio=0.7, val_ratio=0.15):
    """Split temporal data chronologically"""
    n_timestamps = len(temporal_data)
    
    train_end = int(n_timestamps * train_ratio)
    val_end = int(n_timestamps * (train_ratio + val_ratio))
    
    train_data = temporal_data[:train_end]
    val_data = temporal_data[train_end:val_end]
    test_data = temporal_data[val_end:]
    
    return train_data, val_data, test_data

# Test function
def test_temporal_gnn():
    """Test the manual temporal GNN implementation"""
    print("🧪 Testing manual temporal GNN implementation...")
    
    # Create dummy data
    num_nodes = 20
    node_features = 4
    sequence_length = 5
    
    # Create sequence of graphs
    x_sequence = []
    edge_index_sequence = []
    
    for t in range(sequence_length):
        # Random node features
        x = torch.randn(num_nodes, node_features)
        x_sequence.append(x)
        
        # Random edges (ensuring some connectivity)
        num_edges = 30
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_index_sequence.append(edge_index)
    
    # Test TemporalGCN
    model = TemporalGCN(
        node_features=node_features,
        hidden_dim=16,
        output_dim=8
    )
    
    output, hidden = model(x_sequence, edge_index_sequence)
    print(f"✅ TemporalGCN output shape: {output.shape}")
    
    # Test TemporalAnomalyDetector
    anomaly_model = TemporalAnomalyDetector(
        node_features=node_features,
        hidden_dim=16,
        embedding_dim=8
    )
    
    embeddings, node_recon, edge_recon, anomaly_scores = anomaly_model(x_sequence, edge_index_sequence)
    print(f"✅ Anomaly model - embeddings: {embeddings.shape}, scores: {anomaly_scores.shape}")
    
    print("🎉 Manual temporal GNN implementation working!")

if __name__ == "__main__":
    test_temporal_gnn()
