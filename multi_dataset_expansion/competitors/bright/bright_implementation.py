#!/usr/bin/env python3
"""
BRIGHT Implementation - Real-time Fraud Detection
Based on: "BRIGHT - Graph Neural Networks in Real-time Fraud Detection" (CIKM'22)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np

class TwoStageDirectedGraph:
    """Two-Stage Directed Graph transformation for BRIGHT"""
    
    def __init__(self):
        self.historical_edges = []
        self.realtime_edges = []
    
    def transform_graph(self, edge_index, timestamps, current_time):
        """Transform graph into two-stage directed graph"""
        
        # Historical subgraph (edges before current time)
        hist_mask = timestamps < current_time
        historical_edges = edge_index[:, hist_mask]
        
        # Real-time subgraph (edges at current time)
        rt_mask = timestamps == current_time
        realtime_edges = edge_index[:, rt_mask]
        
        return historical_edges, realtime_edges

class LambdaNeuralNetwork(nn.Module):
    """Lambda Neural Network architecture for BRIGHT"""
    
    def __init__(self, node_feat_dim=16, hidden_dim=64, num_heads=4):
        super(LambdaNeuralNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Temporal feature encoder
        self.temporal_encoder = nn.Linear(1, hidden_dim // 4)  # Timestamp encoding
        
        # Graph attention layers for batch inference
        self.gat1 = GATConv(node_feat_dim + hidden_dim // 4, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        
        # Real-time inference components
        self.realtime_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Fraud detection head
        self.fraud_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Batch + real-time features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode_temporal_features(self, timestamps):
        """Encode temporal information"""
        # Normalize timestamps
        if len(timestamps) > 1:
            timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        else:
            timestamps_norm = torch.zeros_like(timestamps)
        
        temporal_emb = F.relu(self.temporal_encoder(timestamps_norm.unsqueeze(-1)))
        return temporal_emb
    
    def batch_inference(self, x, edge_index, timestamps):
        """Batch inference for entity embeddings"""
        
        # Encode temporal features
        temporal_emb = self.encode_temporal_features(timestamps)
        
        # Combine node features with temporal embeddings
        if temporal_emb.size(0) != x.size(0):
            # Handle size mismatch by repeating or truncating
            if temporal_emb.size(0) < x.size(0):
                repeat_factor = (x.size(0) + temporal_emb.size(0) - 1) // temporal_emb.size(0)
                temporal_emb = temporal_emb.repeat(repeat_factor, 1)[:x.size(0)]
            else:
                temporal_emb = temporal_emb[:x.size(0)]
        
        # Concatenate features
        h = torch.cat([x, temporal_emb], dim=1)
        
        # Apply GAT layers
        h = F.relu(self.gat1(h, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        batch_embeddings = self.gat2(h, edge_index)
        
        return batch_embeddings
    
    def realtime_inference(self, batch_embeddings, realtime_features):
        """Real-time inference for transaction prediction"""
        
        # Encode real-time features
        rt_embeddings = F.relu(self.realtime_encoder(realtime_features))
        
        # Combine batch and real-time embeddings
        combined = torch.cat([batch_embeddings, rt_embeddings], dim=1)
        
        # Fraud prediction
        fraud_scores = self.fraud_detector(combined)
        
        return fraud_scores
    
    def forward(self, x, historical_edges, realtime_edges, timestamps):
        """Complete forward pass"""
        
        # Batch inference on historical graph
        batch_embeddings = self.batch_inference(x, historical_edges, timestamps)
        
        # For real-time inference, use mean pooling of batch embeddings as context
        if realtime_edges.size(1) > 0:
            # Get embeddings for real-time nodes
            rt_nodes = torch.unique(realtime_edges.view(-1))
            if len(rt_nodes) > 0:
                rt_batch_emb = batch_embeddings[rt_nodes]
                rt_context = torch.mean(rt_batch_emb, dim=0, keepdim=True)
            else:
                rt_context = torch.mean(batch_embeddings, dim=0, keepdim=True)
        else:
            rt_context = torch.mean(batch_embeddings, dim=0, keepdim=True)
        
        # Real-time inference
        fraud_scores = self.realtime_inference(rt_context, rt_context)
        
        return fraud_scores.squeeze()

class BRIGHTFramework:
    """Complete BRIGHT framework"""
    
    def __init__(self, node_feat_dim=16, hidden_dim=64):
        self.graph_transformer = TwoStageDirectedGraph()
        self.lambda_network = LambdaNeuralNetwork(node_feat_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.lambda_network.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
    
    def fit(self, temporal_data, labels, epochs=20):
        """Train BRIGHT framework"""
        
        self.lambda_network.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for data, label in zip(temporal_data, labels):
                self.optimizer.zero_grad()
                
                x, edge_index, timestamps = data
                current_time = timestamps.max()
                
                # Transform graph
                hist_edges, rt_edges = self.graph_transformer.transform_graph(
                    edge_index, timestamps, current_time
                )
                
                # Forward pass
                pred = self.lambda_network(x, hist_edges, rt_edges, timestamps)
                
                # Handle single prediction vs batch
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                
                loss = self.criterion(pred, torch.tensor([label], dtype=torch.float))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"BRIGHT Epoch {epoch}: Loss = {epoch_loss/len(temporal_data):.4f}")
    
    def predict(self, temporal_data):
        """Predict using BRIGHT framework"""
        
        self.lambda_network.eval()
        scores = []
        
        with torch.no_grad():
            for data in temporal_data:
                x, edge_index, timestamps = data
                current_time = timestamps.max()
                
                # Transform graph
                hist_edges, rt_edges = self.graph_transformer.transform_graph(
                    edge_index, timestamps, current_time
                )
                
                # Predict
                score = self.lambda_network(x, hist_edges, rt_edges, timestamps)
                
                if score.dim() == 0:
                    scores.append(score.item())
                else:
                    scores.append(score.mean().item())
        
        return np.array(scores)

def test_bright():
    """Test BRIGHT implementation"""
    print("🧪 Testing BRIGHT implementation...")
    
    # Create dummy data
    num_nodes = 50
    node_feat_dim = 16
    
    # Dummy temporal data
    x = torch.randn(num_nodes, node_feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    timestamps = torch.randint(0, 10, (100,)).float()
    
    # Test framework
    bright = BRIGHTFramework(node_feat_dim)
    
    # Test prediction
    data = (x, edge_index, timestamps)
    prediction = bright.predict([data])
    
    print(f"✅ BRIGHT output: {prediction}")
    print(f"✅ BRIGHT test completed!")

if __name__ == "__main__":
    test_bright()
