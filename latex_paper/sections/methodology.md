
# 3. Methodology

## 3.1 Problem Formulation

**Temporal Graph Anomaly Detection**: Given a temporal graph $G = \{G_1, G_2, ..., G_T\}$ where $G_t = (V_t, E_t, X_t)$ represents the graph at time $t$ with nodes $V_t$, edges $E_t$, and node features $X_t$, our goal is to identify anomalous nodes in real-time as new interactions arrive.

**Deployment Requirements**: Unlike retrospective analysis, real-time deployment requires:
1. **Early Detection**: Identify anomalies with limited interaction history
2. **Cold Start Handling**: Evaluate new nodes with minimal data
3. **Temporal Consistency**: Maintain stable performance across time periods
4. **Real-time Processing**: Process streaming graph updates efficiently

## 3.2 TempAnom-GNN Architecture

Our architecture integrates three components optimized for deployment scenarios:

### 3.2.1 Temporal Evolution Encoder

**Motivation**: Captures fine-grained temporal patterns that emerge before statistical anomalies become apparent.

**Design**: Building on DyRep [13], we encode temporal evolution through:

$$h_v^{(t)} = \text{GRU}(h_v^{(t-1)}, \text{Aggregate}(\{m_{uv}^{(\tau)} : \tau \leq t\}))$$

where $m_{uv}^{(\tau)}$ represents the message from node $u$ to $v$ at time $\tau$.

**Deployment Advantage**: Temporal patterns often precede statistical anomalies, enabling early detection.

### 3.2.2 Memory Mechanism

**Motivation**: Maintains representations of normal behavior patterns for comparison with current activity.

**Design**: Following TGN [12], we maintain memory states:

$$\text{Memory}_v^{(t)} = \text{Update}(\text{Memory}_v^{(t-1)}, h_v^{(t)}, \text{is\_normal}(G_t))$$

**Deployment Advantage**: Enables detection of deviations from established normal patterns.

### 3.2.3 Trajectory Predictor

**Motivation**: Predicts future interaction patterns to identify anomalous trajectories early.

**Design**: Inspired by JODIE [14], we predict future embeddings:

$$\hat{h}_v^{(t+\Delta)} = \text{MLP}(h_v^{(t)}, \Delta)$$

**Deployment Advantage**: Trajectory deviations signal emerging anomalies before they fully manifest.

### 3.2.4 Unified Anomaly Scoring

**Integration**: We combine components through learned weights:

$$\text{Score}_v^{(t)} = \alpha \cdot \text{EvolutionScore}_v^{(t)} + \beta \cdot \text{MemoryScore}_v^{(t)} + \gamma \cdot \text{TrajectoryScore}_v^{(t)}$$

**Optimization**: Weights are learned through deployment-focused loss functions that prioritize early detection and cold start performance.

## 3.3 Training Strategy

### 3.3.1 Deployment-Focused Loss Function

Traditional anomaly detection optimizes for retrospective performance. We design loss functions specifically for deployment requirements:

$$L = L_{\text{detection}} + \lambda_1 L_{\text{early}} + \lambda_2 L_{\text{coldstart}} + \lambda_3 L_{\text{consistency}}$$

where:
- $L_{\text{detection}}$: Standard anomaly detection loss
- $L_{\text{early}}$: Early detection penalty (higher weight for early time steps)
- $L_{\text{coldstart}}$: Cold start penalty (higher weight for nodes with limited data)
- $L_{\text{consistency}}$: Temporal consistency regularization

### 3.3.2 Temporal Batching Strategy

**Challenge**: Traditional batching ignores temporal dependencies.

**Solution**: We employ temporal batching that preserves chronological order:
1. Sort interactions by timestamp
2. Create temporal windows with overlap
3. Ensure causality (no future information leakage)
4. Maintain memory states across batches

## 3.4 Component Analysis Framework

**Motivation**: Understanding which components contribute most to deployment performance.

**Methodology**: We evaluate single-component variants:
- **Evolution-only**: Temporal encoding without memory or prediction
- **Memory-only**: Memory mechanism without evolution or prediction  
- **Prediction-only**: Trajectory prediction without memory or evolution
- **Full-system**: All components combined

**Evaluation**: Each variant is tested on deployment scenarios (early detection, cold start) to identify optimal architectures.

## 3.5 Design Decisions for Deployment

### 3.5.1 Scalability Considerations

**Memory Efficiency**: Bounded memory states with periodic cleanup
**Computational Efficiency**: Linear complexity in number of nodes
**Update Efficiency**: Incremental updates for streaming data

### 3.5.2 Robustness Measures

**Temporal Stability**: Consistent performance across time periods
**Data Quality**: Handling missing or noisy interaction data
**Adaptation**: Mechanisms for evolving fraud patterns

Our methodology prioritizes real-world deployment requirements while maintaining the representational power needed for effective anomaly detection. The component analysis framework provides actionable insights for practitioners deploying temporal graph methods.
