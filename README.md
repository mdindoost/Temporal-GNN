# TempAnom-GNN: Temporal Graph Neural Networks for Real-time Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-green.svg)](#citation)

> **Deployment-focused temporal graph neural networks for real-time anomaly detection in dynamic networks**

This repository contains the complete implementation for our paper "TempAnom-GNN: Temporal Graph Neural Networks for Real-time Fraud Detection in Dynamic Networks".

## 🎯 Key Research Findings

Our verified experimental results demonstrate:

- **🚀 20.8% improvement** in early detection (95% CI: 0.171-0.246, p < 0.001)
- **⭐ 13.2% improvement** for cold start scenarios (95% CI: 0.055-0.209, p < 0.01)  
- **💡 Component interference discovery**: Evolution-only outperforms complex combinations
- **📊 Verified baselines**: 25.08× separation ratio for statistical methods
- **🔬 Statistical rigor**: 40 experiments with bootstrap validation

## 🏗️ Architecture Overview

```
TempAnom-GNN Components:
┌─────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Bitcoin   │───▶│   Evolution     │───▶│   Memory     │───▶│ Trajectory  │
│   Network   │    │   Component     │    │  Component   │    │ Component   │
│ (24,186     │    │  (Temporal      │    │ (Normal      │    │(Future      │
│  edges)     │    │   Patterns)     │    │  Behavior)   │    │ Prediction) │
└─────────────┘    └─────────────────┘    └──────────────┘    └─────────────┘
                            │                      │                   │
                            └──────────────────────┼───────────────────┘
                                                   ▼
                                          ┌─────────────────┐
                                          │ Anomaly Scoring │
                                          │ α·E + β·M + γ·T │
                                          │ (Configurable)  │
                                          └─────────────────┘
```

## 📊 Verified Performance Results

### Deployment Scenarios (Table 2 - Verified ✅)

| Evaluation Type | Method | Performance | Statistical Significance | Use Case |
|----------------|--------|-------------|-------------------------|----------|
| **Early Detection** | TempAnom-GNN | **+20.8%** | p < 0.001 | Prospective fraud detection |
| **Cold Start** | TempAnom-GNN | **+13.2%** | p < 0.01 | New user evaluation |
| **Retrospective** | Negative Ratio | **25.08× ratio** | Baseline | Historical analysis |

### Component Analysis (Table 5 - Corrected ✅)

| Configuration | Weights | Early Detection | Cold Start | Best For |
|--------------|---------|----------------|------------|----------|
| **Evolution Only** | (1.0, 0.0, 0.0) | **0.300 ± 0.115** | 0.360 ± 0.135 | **Early Detection** |
| **Memory Only** | (0.0, 1.0, 0.0) | 0.260 ± 0.051 | **0.460 ± 0.134** | **Cold Start** |
| **Strong Evolution** | (0.7, 0.2, 0.1) | 0.280 ± 0.089 | 0.340 ± 0.122 | Balanced |
| **Equal Weights** | (0.3, 0.3, 0.3) | 0.200 ± 0.058 | 0.260 ± 0.108 | General use |

### 💡 Key Discovery: Component Interference
**Evolution-only components consistently outperform complex combinations**, challenging conventional assumptions about multi-component architectures.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 4-8GB GPU memory

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mdindoost/temporal-gnn-project.git
cd temporal-gnn-project
```

2. **Set up environment:**
```bash
# Create conda environment
conda create -n temporal-gnn python=3.10
conda activate temporal-gnn

# Install dependencies
pip install torch==2.7.0+cu118 torch-geometric==2.6.1
pip install pandas numpy matplotlib scikit-learn networkx
```

3. **Download datasets:**
```bash
python download_datasets.py
```

### Basic Usage

```python
from temporal_anomaly_detector import TemporalAnomalyDetector

# Load Bitcoin Alpha dataset (verified)
detector = TemporalAnomalyDetector(
    data_path='data/processed/bitcoin_alpha_processed.csv'
)

# Train with verified configuration
detector.train_temporal_model(
    epochs=50,
    learning_rate=0.01,
    alpha=1.0,  # Evolution-only (best for early detection)
    beta=0.0,
    gamma=0.0
)

# Evaluate on deployment scenarios
early_detection_score = detector.evaluate_early_detection()
cold_start_score = detector.evaluate_cold_start()

print(f"Early Detection Score: {early_detection_score:.3f}")
print(f"Cold Start Score: {cold_start_score:.3f}")
```

## 📊 Dataset Information

### Bitcoin Trust Networks (Verified ✅)
- **Source**: Stanford SNAP Bitcoin Alpha dataset
- **Edges**: 24,186 trust relationships
- **Users**: 3,783 unique users
- **Suspicious Users**: 73 (verified ground truth)
- **Ratings**: -1 (distrust) to +1 (trust)
- **Temporal Span**: Multiple years

### Ground Truth Definition (Verified ✅)
```python
# Suspicious user criteria (exact from paper)
def create_ground_truth(df):
    suspicious_users = []
    for user in df['target_idx'].unique():
        user_ratings = df[df['target_idx'] == user]
        if len(user_ratings) >= 5:  # Min 5 interactions
            negative_ratio = sum(user_ratings['rating'] < 0) / len(user_ratings)
            if negative_ratio > 0.3:  # >30% negative ratings
                suspicious_users.append(user)
    return suspicious_users  # Returns 73 users
```

## 🧪 Reproducing Paper Results

### Run Complete Verification Pipeline
```bash
# PHASE 1: Data verification
cd assess_paper_results
python data_verification/ground_truth_analysis.py

# PHASE 2: Baseline verification  
python final_verification_with_exact_methodology.py

# PHASE 3: Component analysis
python fixed_ablation_study.py
```

### Expected Output (Verified ✅)
```
✅ Dataset: 24,186 edges, 3,783 users
✅ Ground truth: 73 suspicious users  
✅ Negative Ratio baseline: 25.08× separation, 0.460 precision@50
✅ TempAnom-GNN: 1.33× separation, 0.670 precision@50
✅ Early detection improvement: +20.8% (p < 0.001)
✅ Cold start improvement: +13.2% (p < 0.01)
```

### Component Configuration Examples
```python
# Best configurations (verified from Table 5)

# For early detection priority
config_early = {
    'alpha': 1.0,  # Evolution only
    'beta': 0.0,
    'gamma': 0.0,
    'expected_score': 0.300  # ± 0.115
}

# For cold start priority  
config_cold = {
    'alpha': 0.0,
    'beta': 1.0,   # Memory only
    'gamma': 0.0,
    'expected_score': 0.460  # ± 0.134
}

# Balanced approach
config_balanced = {
    'alpha': 0.7,   # Strong evolution
    'beta': 0.2,
    'gamma': 0.1,
    'expected_score': 0.280  # ± 0.089
}
```

## 📁 Project Structure

```
temporal-gnn-project/
├── 📊 data/
│   ├── bitcoin/                           # Raw SNAP datasets
│   │   ├── soc-sign-bitcoin-alpha.csv.gz  # Bitcoin Alpha network
│   │   └── soc-sign-bitcoin-otc.csv.gz    # Bitcoin OTC network
│   └── processed/                         # Processed datasets
│       └── bitcoin_alpha_processed.csv    # Verified dataset (24,186 edges)
├── 🔬 assess_paper_results/               # Verification pipeline
│   ├── data_verification/                 # Ground truth verification
│   ├── experiments/                       # Component analysis
│   └── figures/                          # Generated visualizations
├── 🧠 src/                               # Core implementation
│   ├── temporal_anomaly_detector.py      # Main detector class
│   ├── temporal_memory_module.py         # Memory components
│   └── static_gnn_baseline.py           # Baseline methods
├── 📈 results/                           # Experimental outputs
└── 📋 configs/                           # Configuration files
```

## 🔬 Technical Details

### Model Architecture
- **Evolution Component**: Temporal pattern encoding (DyRep-inspired)
- **Memory Component**: Normal behavior modeling (TGN-inspired) 
- **Prediction Component**: Future trajectory prediction (JODIE-inspired)
- **Parameters**: ~150K trainable parameters
- **Training Time**: 2-3 hours (50 epochs)

### Statistical Validation
```python
# Bootstrap validation (as used in paper)
def bootstrap_evaluation(scores, labels, n_samples=30):
    # 95% confidence intervals from 30 bootstrap samples
    results = []
    for i in range(n_samples):
        # Bootstrap sample with replacement
        indices = np.random.choice(len(scores), size=len(scores), replace=True)
        sample_scores = scores[indices]
        sample_labels = labels[indices]
        
        # Calculate metric (e.g., precision@10)
        result = evaluate_metric(sample_scores, sample_labels)
        results.append(result)
    
    # 95% confidence interval
    ci_lower = np.percentile(results, 2.5)
    ci_upper = np.percentile(results, 97.5)
    return np.mean(results), (ci_lower, ci_upper)
```

## 📚 Key Research Insights

### 1. Deployment vs Retrospective Analysis
- **Simple baselines excel** when complete histories available (25.08× separation)
- **Temporal methods shine** in deployment scenarios (+20.8% early detection)
- **Different evaluation needed** for retrospective vs prospective analysis

### 2. Component Interference Phenomenon  
- **Evolution-only** achieves best early detection (0.300 ± 0.115)
- **Memory-only** optimal for cold start (0.460 ± 0.134)
- **Complex combinations underperform** due to component interference

### 3. Statistical Validation Importance
- **Bootstrap confidence intervals** essential for sparse anomaly detection
- **Multiple seeds required** (5 seeds minimum for stable results)
- **Precision@10 evaluation** creates natural variance in imbalanced datasets

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{,
  title={TempAnom-GNN: Temporal Graph Neural Networks for Real-time Fraud Detection in Dynamic Networks},
  author={Dindoost, Mohammad and [Co-authors]},
  booktitle={Proceedings of [Conference]},
  year={2025},
  note={Under Review}
}
```

## 🤝 Contributing

We welcome contributions! Areas for extension:

- **Cross-domain validation**: Social networks, e-commerce platforms
- **Real-time optimization**: Streaming deployment improvements  
- **Alternative ground truth**: Beyond rating-based fraud detection
- **Interpretability tools**: Explainable temporal pattern analysis

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stanford SNAP**: Bitcoin trust network datasets
- **NJIT**: High-performance computing resources
- **PyTorch Geometric**: Graph neural network framework
- **Research Community**: Temporal graph neural network advances

## 📞 Contact

- **Author**: Mohammad Dindoost  
- **Email**: md724@njit.edu
- **Institution**: New Jersey Institute of Technology
- **GitHub**: [@mdindoost](https://github.com/mdindoost)

---

## 🔗 Additional Resources

- 📄 **Paper**: [TempAnom-GNN Paper](paper/main.pdf) *(available upon acceptance)*
- 📊 **Datasets**: [Stanford SNAP Bitcoin Networks](https://snap.stanford.edu/data/)
- 🛠️ **Documentation**: [Implementation Guide](docs/implementation.md)
- 🧪 **Experiments**: [Verification Results](assess_paper_results/)

---

**Status**: ✅ **Research Complete** | 📝 **Paper Under Review** | 🔬 **Results Verified** | 🚀 **Ready for Use**

*Advancing temporal graph anomaly detection from research to real-world deployment.* 🌐🧠⚡
