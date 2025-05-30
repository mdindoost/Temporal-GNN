# Temporal GNN Anomaly Detection - Datasets

## Dataset Overview

### 1. Bitcoin Trust Networks
- **Source**: Stanford SNAP
- **Description**: Signed trust networks from Bitcoin trading platforms
- **Files**:
  - `bitcoin/soc-sign-bitcoin-alpha.csv.gz`: Bitcoin Alpha platform
  - `bitcoin/soc-sign-bitcoin-otc.csv.gz`: Bitcoin OTC platform
- **Format**: [source_user, target_user, rating, timestamp]
- **Ratings**: +1 (trust), -1 (distrust)
- **Use case**: Financial fraud detection, trust relationship anomalies

### 2. Synthetic Temporal Graph
- **Source**: Generated
- **Description**: Synthetic temporal graph with injected anomalies
- **Files**:
  - `synthetic/temporal_graph_with_anomalies.pkl`: Full temporal data
  - `synthetic/temporal_graph_summary.csv`: Summary statistics
- **Anomalies**:
  - Timestamp 15: Star burst (one node connects to many)
  - Timestamp 30: Dense clique formation
  - Timestamp 45: Network disconnection
- **Use case**: Algorithm development and testing

### 3. Data Processing
- **Processed files**: `processed/` directory contains cleaned datasets
- **Format**: Standardized CSV files ready for GNN training

## Usage Notes
- All timestamps are Unix epoch time
- Node IDs are integers (may need remapping for GNNs)
- Negative ratings in Bitcoin data can indicate potential fraud
- Synthetic data has ground truth anomaly labels for evaluation

