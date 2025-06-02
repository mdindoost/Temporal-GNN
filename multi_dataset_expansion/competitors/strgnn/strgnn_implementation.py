#!/usr/bin/env python3
"""
StrGNN Implementation - Competitor Method
Based on: "Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs" (CIKM'21)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool
import numpy as np

class StrGNN(nn.Module):
    """
    Structural Temporal Graph Neural Network
    Implements the paper methodology: h-hop subgraph + GCN + GRU
    """
    
    def __init__(self, node_feat_dim=16, hidden_dim=64, k_hop=2):
        super(StrGNN, self).__init__()
        
        self.k_hop = k_hop
        self.hidden_dim = hidden_dim
        
        # Node labeling dimension (distance-based labeling)
        self.label_dim = 3
        
        # GCN layers for structural embedding
        self.gcn1 = GCNConv(node_feat_dim + self.label_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Sort pooling for fixed-size representation
        self.pool_ratio = 0.5
        
        # Temporal GRU layers
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Anomaly detection head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def extract_h_hop_subgraph(self, edge_index, center_nodes, num_nodes):
        """Extract h-hop enclosing subgraph around center nodes"""
        
        # For simplicity, use ego-networks around center nodes
        subgraph_nodes = set(center_nodes)
        
        for hop in range(self.k_hop):
            new_nodes = set()
            for node in subgraph_nodes:
                # Find neighbors
                neighbors_out = edge_index[1][edge_index[0] == node]
                neighbors_in = edge_index[0][edge_index[1] == node]
                new_nodes.update(neighbors_out.tolist())
                new_nodes.update(neighbors_in.tolist())
            
            subgraph_nodes.update(new_nodes)
        
        subgraph_nodes = list(subgraph_nodes)
        
        # Create node mapping
        node_mapping = {old: new for new, old in enumerate(subgraph_nodes)}
        
        # Extract subgraph edges
        mask = torch.isin(edge_index[0], torch.tensor(subgraph_nodes)) & \
               torch.isin(edge_index[1], torch.tensor(subgraph_nodes))
        
        subgraph_edges = edge_index[:, mask]
        
        # Remap to local indices
        subgraph_edges[0] = torch.tensor([node_mapping[n.item()] for n in subgraph_edges[0]])
        subgraph_edges[1] = torch.tensor([node_mapping[n.item()] for n in subgraph_edges[1]])
        
        return subgraph_nodes, subgraph_edges, node_mapping
    
    def create_node_labels(self, subgraph_nodes, center_nodes):
        """Create distance-based node labels"""
        
        labels = torch.zeros(len(subgraph_nodes), self.label_dim)
        
        for i, node in enumerate(subgraph_nodes):
            if node in center_nodes:
                labels[i] = torch.tensor([1, 0, 0])  # Center node
            else:
                # Distance-based labeling (simplified)
                labels[i] = torch.tensor([0, 1, 0])  # Non-center
        
        return labels
    
    def forward_snapshot(self, x, edge_index, target_nodes):
        """Process single temporal snapshot"""
        
        # Extract subgraph around target nodes
        subgraph_nodes, subgraph_edges, node_mapping = self.extract_h_hop_subgraph(
            edge_index, target_nodes, x.size(0)
        )
        
        if len(subgraph_nodes) == 0:
            return torch.zeros(1, self.hidden_dim)
        
        # Get subgraph features
        subgraph_x = x[subgraph_nodes]
        
        # Add node labels
        node_labels = self.create_node_labels(subgraph_nodes, target_nodes)
        subgraph_x_labeled = torch.cat([subgraph_x, node_labels], dim=1)
        
        # Apply GCN layers
        h = F.relu(self.gcn1(subgraph_x_labeled, subgraph_edges))
        h = F.relu(self.gcn2(h, subgraph_edges))
        
        # Sort pooling for fixed-size representation
        k = max(1, int(len(subgraph_nodes) * self.pool_ratio))
        pooled_h = global_sort_pool(h, batch=None, k=k)
        
        # Ensure fixed size
        if pooled_h.size(0) < self.hidden_dim:
            padding = torch.zeros(self.hidden_dim - pooled_h.size(0))
            pooled_h = torch.cat([pooled_h, padding])
        else:
            pooled_h = pooled_h[:self.hidden_dim]
        
        return pooled_h.unsqueeze(0)
    
    def forward(self, temporal_data):
        """
        Forward pass for temporal sequence
        temporal_data: List of (x, edge_index, target_nodes) for each timestamp
        """
        
        # Process each snapshot
        temporal_embeddings = []
        
        for x, edge_index, target_nodes in temporal_data:
            snapshot_emb = self.forward_snapshot(x, edge_index, target_nodes)
            temporal_embeddings.append(snapshot_emb)
        
        if len(temporal_embeddings) == 1:
            final_emb = temporal_embeddings[0]
        else:
            # Stack and process with GRU
            temporal_seq = torch.cat(temporal_embeddings, dim=0).unsqueeze(0)  # [1, T, H]
            gru_out, _ = self.gru(temporal_seq)
            final_emb = gru_out[0, -1, :].unsqueeze(0)  # Last timestep
        
        # Classify
        anomaly_score = self.classifier(final_emb)
        
        return anomaly_score.squeeze()

class StrGNNWrapper:
    """Wrapper to integrate StrGNN with your evaluation pipeline"""
    
    def __init__(self, node_feat_dim=16, hidden_dim=64):
        self.model = StrGNN(node_feat_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
    def fit(self, temporal_data, labels, epochs=20):
        """Train StrGNN on temporal data"""
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for data, label in zip(temporal_data, labels):
                self.optimizer.zero_grad()
                
                # Forward pass
                pred = self.model(data)
                loss = self.criterion(pred.unsqueeze(0), torch.tensor([label], dtype=torch.float))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"StrGNN Epoch {epoch}: Loss = {epoch_loss/len(temporal_data):.4f}")
    
    def predict(self, temporal_data):
        """Predict anomaly scores"""
        
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for data in temporal_data:
                score = self.model(data)
                scores.append(score.item())
        
        return np.array(scores)

def test_strgnn():
    """Test StrGNN implementation"""
    print("🧪 Testing StrGNN implementation...")
    
    # Create dummy data
    num_nodes = 50
    node_feat_dim = 16
    
    # Dummy temporal data
    temporal_data = []
    for t in range(3):  # 3 timestamps
        x = torch.randn(num_nodes, node_feat_dim)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        target_nodes = [0, 1, 2]  # Focus on these nodes
        temporal_data.append((x, edge_index, target_nodes))
    
    # Test model
    model = StrGNN(node_feat_dim)
    output = model([temporal_data])  # Single sequence
    
    print(f"✅ StrGNN output shape: {output.shape}")
    print(f"✅ StrGNN test completed!")

if __name__ == "__main__":
    test_strgnn()
