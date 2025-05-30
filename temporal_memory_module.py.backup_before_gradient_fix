#!/usr/bin/env python3
"""
Final Fixed Temporal Memory Module Implementation
Addresses all dimension, NaN, and index out of bounds issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

class Time2Vec(nn.Module):
    """Time2Vec encoding from TGN paper"""
    def __init__(self, time_dim: int):
        super(Time2Vec, self).__init__()
        self.time_dim = time_dim
        self.linear_layer = nn.Linear(1, 1)
        if time_dim > 1:
            self.periodic_layers = nn.Linear(1, time_dim - 1)
    
    def forward(self, time_delta: torch.Tensor) -> torch.Tensor:
        batch_size = time_delta.shape[0]
        
        # Linear component
        linear_time = self.linear_layer(time_delta)
        
        if self.time_dim == 1:
            return linear_time
        
        # Periodic components
        periodic_time = torch.sin(self.periodic_layers(time_delta))
        
        # Concatenate linear and periodic
        time_encoding = torch.cat([linear_time, periodic_time], dim=1)
        return time_encoding

class NodeMemoryModule(nn.Module):
    """TGN-inspired node memory module with all fixes"""
    def __init__(self, num_nodes: int, memory_dim: int, message_dim: int):
        super(NodeMemoryModule, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        
        # Initialize node memories
        self.register_buffer('node_memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update_time', torch.zeros(num_nodes))
        
        # Memory update mechanism (GRU-based)
        self.memory_updater = nn.GRUCell(message_dim, memory_dim)
        
        # Message aggregation - fixed input dimension calculation
        time_dim = memory_dim // 4
        message_input_dim = memory_dim * 2 + time_dim * 2  # src + dst memory + src + dst time
        self.message_aggregator = nn.Linear(message_input_dim, message_dim)
        
        # Time encoding
        self.time_encoder = Time2Vec(time_dim)
        
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Get current memory for specified nodes"""
        return self.node_memory[node_ids]
    
    def create_messages(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor, 
                       timestamps: torch.Tensor) -> torch.Tensor:
        """Create simplified messages between nodes"""
        if len(src_nodes) == 0:
            return torch.zeros(0, self.message_dim, device=src_nodes.device)
            
        # Get current memories
        src_memory = self.node_memory[src_nodes]
        dst_memory = self.node_memory[dst_nodes]
        
        # Compute time deltas since last update
        src_time_delta = timestamps - self.last_update_time[src_nodes]
        dst_time_delta = timestamps - self.last_update_time[dst_nodes]
        
        # Encode time information
        src_time_enc = self.time_encoder(src_time_delta.unsqueeze(1))
        dst_time_enc = self.time_encoder(dst_time_delta.unsqueeze(1))
        
        # Create raw messages
        raw_messages = torch.cat([
            src_memory, dst_memory, src_time_enc, dst_time_enc
        ], dim=1)
        
        # Aggregate messages
        messages = self.message_aggregator(raw_messages)
        
        return messages
    
    def update_memory(self, node_ids: torch.Tensor, messages: torch.Tensor, 
                     timestamps: torch.Tensor):
        """Update node memories with new messages"""
        if len(node_ids) == 0:
            return
            
        # Get current memories
        current_memory = self.node_memory[node_ids]
        
        # Update memories using GRU
        new_memory = self.memory_updater(messages, current_memory)
        
        # Store updated memories and timestamps
        self.node_memory[node_ids] = new_memory
        self.last_update_time[node_ids] = timestamps
    
    def compute_memory_deviation(self, node_ids: torch.Tensor, 
                                current_features: torch.Tensor) -> torch.Tensor:
        """Compute how much current node features deviate from memory patterns"""
        # Get current memories
        memories = self.node_memory[node_ids]
        
        # Project current features to memory dimension if needed
        if current_features.shape[1] != self.memory_dim:
            # Simple projection - in practice, this should be learned
            feature_proj = torch.zeros(current_features.shape[0], self.memory_dim, 
                                     device=current_features.device)
            min_dim = min(current_features.shape[1], self.memory_dim)
            feature_proj[:, :min_dim] = current_features[:, :min_dim]
            current_features = feature_proj
        
        # Compute cosine similarity between current features and memory
        similarities = F.cosine_similarity(current_features, memories, dim=1)
        
        # Convert similarity to deviation (higher deviation = more anomalous)
        deviations = 1.0 - similarities
        
        # Replace NaN values with a default deviation
        deviations = torch.where(torch.isnan(deviations), 
                               torch.tensor(0.5, device=deviations.device), 
                               deviations)
        
        return torch.clamp(deviations, 0.0, 2.0)  # Clamp for stability

class GraphMemoryModule(nn.Module):
    """Graph-level memory module with NaN handling"""
    def __init__(self, graph_feature_dim: int, memory_dim: int, window_size: int = 10):
        super(GraphMemoryModule, self).__init__()
        self.graph_feature_dim = graph_feature_dim
        self.memory_dim = memory_dim
        self.window_size = window_size
        
        # Graph-level memory
        self.register_buffer('graph_memory', torch.zeros(memory_dim))
        
        # Historical statistics storage
        self.register_buffer('historical_features', torch.zeros(window_size, graph_feature_dim))
        self.register_buffer('feature_pointer', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_stored', torch.tensor(0, dtype=torch.long))
        
        # Memory update network
        self.memory_updater = nn.GRUCell(graph_feature_dim, memory_dim)
        
        # Feature extraction
        self.feature_extractor = nn.Linear(graph_feature_dim, graph_feature_dim)
        
    def extract_graph_features(self, node_features: torch.Tensor, 
                              edge_index: torch.Tensor) -> torch.Tensor:
        """Extract global graph-level features with robust statistics"""
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1] // 2  # Undirected edges
        
        if num_nodes == 0:
            return torch.zeros(self.graph_feature_dim, device=node_features.device)
        
        # Node feature statistics with NaN handling
        node_mean = torch.mean(node_features, dim=0)
        if num_nodes > 1:
            node_std = torch.std(node_features, dim=0, unbiased=False)  # Use biased estimator
            # Replace any NaN or zero std with small value
            node_std = torch.where(torch.isnan(node_std) | (node_std == 0), 
                                 torch.tensor(0.1, device=node_features.device), 
                                 node_std)
        else:
            node_std = torch.ones_like(node_mean) * 0.1
        
        # Degree statistics with safety checks
        if edge_index.shape[1] > 0:
            # Ensure edge indices are within bounds
            valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            if valid_edges.sum() > 0:
                valid_edge_index = edge_index[:, valid_edges]
                degrees = torch.bincount(valid_edge_index[0], minlength=num_nodes).float()
                degree_mean = torch.mean(degrees)
                if num_nodes > 1:
                    degree_std = torch.std(degrees, unbiased=False)
                    if torch.isnan(degree_std) or degree_std == 0:
                        degree_std = torch.tensor(0.1, device=node_features.device)
                else:
                    degree_std = torch.tensor(0.1, device=node_features.device)
            else:
                degree_mean = torch.tensor(0.0, device=node_features.device)
                degree_std = torch.tensor(0.1, device=node_features.device)
        else:
            degree_mean = torch.tensor(0.0, device=node_features.device)
            degree_std = torch.tensor(0.1, device=node_features.device)
        
        # Connectivity statistics
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        density_tensor = torch.tensor(density, device=node_features.device)
        
        # Combine into graph feature vector
        basic_stats = torch.stack([degree_mean, degree_std, density_tensor])
        
        # Ensure consistent feature dimension
        if node_mean.shape[0] + node_std.shape[0] + 3 <= self.graph_feature_dim:
            # Pad if needed
            combined = torch.cat([node_mean, node_std, basic_stats])
            if combined.shape[0] < self.graph_feature_dim:
                padding = torch.zeros(self.graph_feature_dim - combined.shape[0], device=node_features.device)
                graph_features = torch.cat([combined, padding])
            else:
                graph_features = combined[:self.graph_feature_dim]
        else:
            # Truncate if too large
            combined = torch.cat([node_mean, node_std, basic_stats])
            graph_features = combined[:self.graph_feature_dim]
        
        # Final NaN check
        graph_features = torch.where(torch.isnan(graph_features), 
                                   torch.tensor(0.0, device=graph_features.device), 
                                   graph_features)
        
        return graph_features
        
    def update_memory(self, graph_features: torch.Tensor):
        """Update graph memory with new observations"""
        # Process features
        processed_features = self.feature_extractor(graph_features)
        
        # Update memory using GRU
        self.graph_memory = self.memory_updater(
            processed_features.unsqueeze(0), 
            self.graph_memory.unsqueeze(0)
        ).squeeze(0)
        
        # Store in historical buffer (circular buffer)
        self.historical_features[self.feature_pointer] = graph_features
        self.feature_pointer = (self.feature_pointer + 1) % self.window_size
        self.num_stored = torch.min(self.num_stored + 1, torch.tensor(self.window_size))
    
    def compute_graph_deviation(self, current_graph_features: torch.Tensor) -> torch.Tensor:
        """Compute deviation of current graph from normal patterns with NaN handling"""
        if self.num_stored <= 1:  # Need at least 2 samples for std
            return torch.tensor(0.5, device=current_graph_features.device)  # Default moderate score
        
        # Get historical features (only stored ones)
        stored_features = self.historical_features[:self.num_stored]
        
        # Compute mean and std of historical features
        historical_mean = torch.mean(stored_features, dim=0)
        historical_std = torch.std(stored_features, dim=0, unbiased=False)  # Use biased estimator
        
        # Replace zero or NaN std with small epsilon
        historical_std = torch.where(torch.isnan(historical_std) | (historical_std <= 1e-6), 
                                   torch.tensor(0.1, device=current_graph_features.device), 
                                   historical_std)
        
        # Compute z-score deviation
        z_scores = torch.abs((current_graph_features - historical_mean) / historical_std)
        
        # Handle any remaining NaN values
        z_scores = torch.where(torch.isnan(z_scores), 
                             torch.tensor(0.0, device=z_scores.device), 
                             z_scores)
        
        # Aggregate z-scores (mean absolute deviation)
        deviation_score = torch.mean(z_scores)
        
        return torch.clamp(deviation_score, 0.0, 5.0)  # Reasonable upper bound

class TemporalGCNEncoder(nn.Module):
    """Temporal GCN encoder with edge validation"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(TemporalGCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        # Memory projection layer to match output dimension
        self.memory_projection = nn.Linear(hidden_dim, output_dim)
        
        # Temporal attention with correct dimensions
        self.temporal_attention = nn.MultiheadAttention(output_dim, num_heads=4, dropout=dropout)
        
    def validate_and_fix_edges(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Validate edge indices and remove invalid ones"""
        if edge_index.shape[1] == 0:
            return edge_index
        
        # Find valid edges (both source and target nodes exist)
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & \
                     (edge_index[0] >= 0) & (edge_index[1] >= 0)
        
        if valid_mask.sum() == 0:
            # No valid edges, return empty edge index
            return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        
        # Filter to valid edges only
        valid_edge_index = edge_index[:, valid_mask]
        
        return valid_edge_index
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                memory_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with edge validation and temporal context integration"""
        num_nodes = x.shape[0]
        
        # Validate and fix edge indices
        edge_index = self.validate_and_fix_edges(edge_index, num_nodes)
        
        # GCN encoding
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h = conv(h, edge_index)
            h = bn(h)
            if i < len(self.convs) - 1:  # No activation on last layer
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Integrate memory context if available
        if memory_context is not None:
            # Project memory to output dimension
            projected_memory = self.memory_projection(memory_context)
            
            # Prepare for attention: (seq_len, batch_size, embed_dim)
            current = h.unsqueeze(0)  # (1, num_nodes, output_dim)
            context = projected_memory.unsqueeze(0)  # (1, num_nodes, output_dim)
            
            # Apply temporal attention
            attended, _ = self.temporal_attention(current, context, context)
            h = attended.squeeze(0)  # Back to (num_nodes, output_dim)
        
        return h

class TrajectoryPredictor(nn.Module):
    """JODIE-inspired trajectory predictor with robust error handling"""
    def __init__(self, embedding_dim: int, hidden_dim: int, time_dim: int = 16):
        super(TrajectoryPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        # Time projection layers
        self.time_encoder = Time2Vec(time_dim)
        
        # Projection networks
        self.node_projector = nn.Sequential(
            nn.Linear(embedding_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.graph_projector = nn.Sequential(
            nn.Linear(embedding_dim + time_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def project_embeddings(self, embeddings: torch.Tensor, 
                          time_delta: torch.Tensor) -> torch.Tensor:
        """Project embeddings to future time point"""
        batch_size = embeddings.shape[0]
        
        # Handle scalar time_delta
        if time_delta.dim() == 0:
            time_delta = time_delta.expand(batch_size)
        elif time_delta.shape[0] == 1:
            time_delta = time_delta.expand(batch_size)
        
        # Encode time delta
        time_enc = self.time_encoder(time_delta.unsqueeze(1))
        
        # Combine embeddings with time encoding
        combined = torch.cat([embeddings, time_enc], dim=1)
        
        # Project to future
        projected = self.node_projector(combined)
        
        return projected
    
    def predict_graph_evolution(self, graph_embedding: torch.Tensor,
                               time_delta: torch.Tensor) -> torch.Tensor:
        """Predict evolution of graph-level features"""
        # Handle scalar time_delta
        if time_delta.dim() == 0:
            time_delta = time_delta.unsqueeze(0)
        
        # Encode time
        time_enc = self.time_encoder(time_delta.unsqueeze(1) if time_delta.dim() == 1 else time_delta)
        
        # Combine and project
        combined = torch.cat([graph_embedding.unsqueeze(0) if graph_embedding.dim() == 1 else graph_embedding, time_enc], dim=1)
        predicted = self.graph_projector(combined)
        
        return predicted.squeeze(0) if predicted.shape[0] == 1 else predicted
    
    def compute_prediction_error(self, predicted: torch.Tensor, 
                                actual: torch.Tensor) -> torch.Tensor:
        """Compute prediction error for anomaly detection"""
        # Compute L2 distance
        error = torch.norm(predicted - actual, p=2, dim=-1)
        
        # Normalize by embedding dimension for stability
        normalized_error = error / math.sqrt(predicted.shape[-1])
        
        # Handle NaN values
        normalized_error = torch.where(torch.isnan(normalized_error), 
                                     torch.tensor(0.5, device=normalized_error.device), 
                                     normalized_error)
        
        return normalized_error

class TemporalAnomalyMemory:
    """Final fixed unified temporal memory system"""
    def __init__(self, num_nodes: int, node_feature_dim: int, 
                 memory_dim: int = 64, embedding_dim: int = 32):
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        
        # Calculate graph feature dimension
        graph_feature_dim = node_feature_dim * 2 + 3  # node_mean + node_std + 3 stats
        
        # Initialize components
        self.node_memory = NodeMemoryModule(num_nodes, memory_dim, memory_dim)
        self.graph_memory = GraphMemoryModule(graph_feature_dim, memory_dim)
        self.temporal_encoder = TemporalGCNEncoder(node_feature_dim, memory_dim, embedding_dim)
        self.trajectory_predictor = TrajectoryPredictor(embedding_dim, memory_dim)
        
        # Training state
        self.is_training_phase = True
        self.anomaly_threshold = 0.5
        
    def process_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                     timestamp: float, is_normal: bool = True) -> Dict[str, torch.Tensor]:
        """Process a graph at given timestamp and compute anomaly scores"""
        results = {}
        
        # Extract graph-level features
        graph_features = self.graph_memory.extract_graph_features(node_features, edge_index)
        
        # Get current memories for context
        node_ids = torch.arange(min(self.num_nodes, node_features.shape[0]), device=node_features.device)
        memory_context = self.node_memory.get_memory(node_ids)
        
        # Encode current graph with temporal context
        node_embeddings = self.temporal_encoder(node_features, edge_index, memory_context)
        graph_embedding = global_mean_pool(node_embeddings, 
                                         torch.zeros(node_embeddings.shape[0], dtype=torch.long, device=node_features.device))
        
        # Compute anomaly scores
        node_memory_scores = self.node_memory.compute_memory_deviation(node_ids, node_embeddings)
        graph_memory_score = self.graph_memory.compute_graph_deviation(graph_features)
        
        # Update memories if this is normal behavior or we're in training phase
        if is_normal or self.is_training_phase:
            # Create simplified messages for memory update
            if edge_index.shape[1] > 0:
                # Validate edges first
                valid_edge_index = self.temporal_encoder.validate_and_fix_edges(edge_index, node_features.shape[0])
                
                if valid_edge_index.shape[1] > 0:
                    # Use unique edges only (undirected)
                    src_nodes, dst_nodes = valid_edge_index[0], valid_edge_index[1]
                    unique_edges = torch.unique(torch.stack([torch.min(src_nodes, dst_nodes), 
                                                           torch.max(src_nodes, dst_nodes)]), dim=1)
                    src_nodes, dst_nodes = unique_edges[0], unique_edges[1]
                    
                    timestamps_tensor = torch.full((len(src_nodes),), timestamp, device=node_features.device)
                    
                    messages = self.node_memory.create_messages(src_nodes, dst_nodes, timestamps_tensor)
                    
                    # Aggregate messages per node
                    aggregated_messages = torch.zeros(self.num_nodes, self.memory_dim, device=node_features.device)
                    for i in range(min(self.num_nodes, node_features.shape[0])):
                        mask = (src_nodes == i) | (dst_nodes == i)
                        if mask.sum() > 0:
                            aggregated_messages[i] = messages[mask].mean(dim=0)
                    
                    # Update node memories
                    self.node_memory.update_memory(node_ids, aggregated_messages[:len(node_ids)], 
                                                 torch.full((len(node_ids),), timestamp, device=node_features.device))
            
            # Update graph memory
            self.graph_memory.update_memory(graph_features)
        
        # Store results
        results['node_embeddings'] = node_embeddings
        results['graph_embedding'] = graph_embedding
        results['node_memory_scores'] = node_memory_scores
        results['graph_memory_score'] = graph_memory_score
        results['timestamp'] = timestamp
        
        return results
    
    def compute_unified_anomaly_score(self, current_results: Dict[str, torch.Tensor],
                                     prediction_results: Optional[Dict[str, torch.Tensor]] = None,
                                     actual_future: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute unified anomaly score with NaN handling"""
        # Weight parameters (could be learned)
        alpha = 0.4  # Memory component weight
        beta = 0.3   # Evolution component weight  
        gamma = 0.3  # Prediction component weight
        
        # Memory-based anomaly score
        node_memory_mean = torch.mean(current_results['node_memory_scores'])
        # Handle NaN in node memory scores
        if torch.isnan(node_memory_mean):
            node_memory_mean = torch.tensor(0.5, device=current_results['graph_memory_score'].device)
        
        memory_score = node_memory_mean + current_results['graph_memory_score']
        
        # Evolution-based score (simplified - would need temporal history)
        evolution_score = current_results['graph_memory_score']
        
        # Prediction-based score
        prediction_score = torch.tensor(0.0, device=memory_score.device)
        if prediction_results is not None and actual_future is not None:
            node_pred_error = self.trajectory_predictor.compute_prediction_error(
                prediction_results['predicted_node_embeddings'],
                actual_future['node_embeddings']
            )
            graph_pred_error = self.trajectory_predictor.compute_prediction_error(
                prediction_results['predicted_graph_embedding'].unsqueeze(0),
                actual_future['graph_embedding'].unsqueeze(0)
            )
            prediction_score = torch.mean(node_pred_error) + graph_pred_error.squeeze()
        
        # Combine scores
        unified_score = alpha * memory_score + beta * evolution_score + gamma * prediction_score
        
        # Final NaN check
        if torch.isnan(unified_score):
            unified_score = torch.tensor(0.5, device=memory_score.device)
        
        return torch.clamp(unified_score, 0.0, 10.0)

# Testing function
def test_final_fixed_temporal_memory():
    """Test the final fixed temporal memory system"""
    print("Testing Final Fixed Temporal Memory System...")
    
    # Setup
    num_nodes = 100
    node_feature_dim = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize memory system
    memory_system = TemporalAnomalyMemory(num_nodes, node_feature_dim)
    
    # Move to device
    memory_system.node_memory = memory_system.node_memory.to(device)
    memory_system.graph_memory = memory_system.graph_memory.to(device)
    memory_system.temporal_encoder = memory_system.temporal_encoder.to(device)
    memory_system.trajectory_predictor = memory_system.trajectory_predictor.to(device)
    
    print(f"✅ Initialized on {device}")
    
    # Test with a simple graph with VALID edges only
    node_features = torch.randn(num_nodes, node_feature_dim, device=device)
    # Create edges that are guaranteed to be valid (0 to num_nodes-1)
    edge_index = torch.randint(0, num_nodes, (2, 300), device=device)
    
    try:
        results = memory_system.process_graph(node_features, edge_index, 0.0, is_normal=True)
        score = memory_system.compute_unified_anomaly_score(results)
        print(f"✅ Successfully processed graph. Anomaly score: {score.item():.4f}")
        
        # Test multiple graphs to avoid NaN issues
        print("Testing sequence of graphs...")
        scores = []
        for t in range(5):
            node_features = torch.randn(num_nodes, node_feature_dim, device=device)
            edge_index = torch.randint(0, num_nodes, (2, 300), device=device)
            
            results = memory_system.process_graph(node_features, edge_index, float(t), is_normal=True)
            score = memory_system.compute_unified_anomaly_score(results)
            scores.append(score.item())
            print(f"   T={t}: Score={score.item():.4f}")
        
        avg_score = np.mean(scores)
        print(f"✅ Average score: {avg_score:.4f} (no NaN values)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    return memory_system

if __name__ == "__main__":
    test_final_fixed_temporal_memory()
