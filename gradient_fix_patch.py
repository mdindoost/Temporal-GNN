
# Quick patch for temporal_anomaly_detector.py
# Add this at the beginning of temporal_training_step method

def temporal_training_step(self, graph_data: Data, timestamp: float, 
                          is_normal: bool = True) -> float:
    """Single training step for temporal components with gradient safety"""
    
    # Disable gradient tracking for memory updates
    with torch.no_grad():
        # Process graph through temporal memory (no gradients for memory updates)
        results = self.temporal_memory.process_graph(
            graph_data.x.detach(), graph_data.edge_index.detach(), timestamp, is_normal
        )
    
    # Re-enable gradients for the actual training
    self.optimizer.zero_grad()
    
    # Forward pass with gradients enabled
    node_features = graph_data.x.requires_grad_(True)
    node_ids = torch.arange(min(self.temporal_memory.num_nodes, node_features.shape[0]), 
                           device=node_features.device)
    memory_context = self.temporal_memory.node_memory.get_memory(node_ids).detach()
    
    # Encode with temporal context (gradients enabled)
    node_embeddings = self.temporal_memory.temporal_encoder(
        node_features, graph_data.edge_index, memory_context
    )
    
    if is_normal:
        # Normal graphs should have low anomaly scores
        target_score = torch.tensor(0.1, device=node_embeddings.device)
        # Simple loss based on embedding norm
        current_score = torch.mean(torch.norm(node_embeddings, p=2, dim=1))
        consistency_loss = F.mse_loss(current_score, target_score)
    else:
        consistency_loss = torch.tensor(0.0, device=node_embeddings.device)
    
    # Embedding regularization
    embedding_reg = 0.001 * torch.norm(node_embeddings, p=2)
    
    # Total loss
    total_loss = consistency_loss + embedding_reg
    
    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.get_temporal_parameters(), 1.0)
    self.optimizer.step()
    
    return total_loss.item()
