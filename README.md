# TempAnom-GNN: Temporal Graph Neural Networks for Real-time Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TempAnom-GNN** is a novel temporal graph neural network architecture for real-time anomaly detection in evolving networks. This repository contains the complete implementation, evaluation pipeline, and reproducible results for our multi-dataset validation across financial and social networks.

## 🎯 Key Contributions

- **Component Interference Phenomenon**: Evolution-only components consistently outperform complex combinations
- **Multi-Domain Validation**: Validated across financial (Bitcoin) and social (Wikipedia) networks  
- **Cross-Dataset Consistency**: 25× separation ratio reproduced across datasets (3.6% variance)
- **Deployment-Focused Design**: Real-time performance optimization vs. retrospective accuracy

## 📊 Main Results

| Dataset | TempAnom-GNN | TGN | StrGNN | BRIGHT | Baseline |
|---------|--------------|-----|--------|--------|----------|
| Bitcoin Alpha | 0.750 AUC | 0.820 | 0.880 | 0.720 | **25.08× Sep** |
| Bitcoin OTC | 0.740 AUC | 0.810 | 0.870 | 0.710 | **25.97× Sep** |
| TGB Wiki | 0.730 AUC | 0.780 | 0.850 | 0.690 | 2.10× Sep |

**Key Finding**: Evolution-only component dominance validated across all domains with consistent cross-dataset reproducibility.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- Access to SLURM cluster (for full experiments)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/mdindoost/Temporal-GNN
cd temporal-gnn-project

# Create and activate environment
module load Python/3.10.8  # On HPC systems
python -m venv temporal-gnn-env
source temporal-gnn-env/bin/activate

# Install dependencies
pip install torch==2.7.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1
pip install pandas numpy scikit-learn matplotlib seaborn networkx
pip install torch-geometric-temporal py-tgb  # For TGB datasets
```

### 2. Data Setup

```bash
# Download and process datasets
python download_datasets.py

# Verify data integrity
python -c "
import pandas as pd
import os

datasets = [
    'data/processed/bitcoin_alpha_processed.csv',
    'data/processed/bitcoin_otc_processed.csv'
]

for dataset in datasets:
    if os.path.exists(dataset):
        df = pd.read_csv(dataset)
        print(f'✅ {dataset}: {len(df)} edges')
    else:
        print(f'❌ {dataset}: Not found')
"
```

### 3. Run Complete Evaluation

#### Option A: Full Multi-Dataset Expansion (Recommended)

```bash
# Navigate to expansion directory
cd multi_dataset_expansion

# Submit SLURM job (HPC)
sbatch run_tgb_expansion.slurm

# OR run interactively
source ../set_temporal_gnn
bash run_tgb_expansion.slurm

# Monitor progress
squeue -u $USER
tail -f ../logs/tgb_expansion_*.out
```

#### Option B: Quick Verification (5 minutes)

```bash
# Test core functionality
python temporal_anomaly_detector.py

# Quick baseline comparison
python bitcoin_baseline_comparison.py

# Component analysis
python assess_paper_results/fixed_ablation_study.py
```

### 4. View Results

```bash
# Navigate to results
cd multi_dataset_expansion/results

# View comprehensive results
cat publication_summary.md

# Check generated files
ls -la
# Expected output:
# ✅ final_expansion_results.json
# ✅ final_summary.csv  
# ✅ component_analysis.json
# ✅ tables/comprehensive_results_table.tex
```

## 📁 Repository Structure

```
temporal-gnn-project/
├── data/                                    # Datasets
│   ├── processed/                          # Clean datasets
│   │   ├── bitcoin_alpha_processed.csv     # 24,186 edges, 73 suspicious users
│   │   └── bitcoin_otc_processed.csv       # 35,592 edges, 219 suspicious users
│   └── bitcoin/                            # Raw SNAP datasets
├── multi_dataset_expansion/                 # Complete evaluation pipeline
│   ├── results/                            # Generated results
│   │   ├── final_expansion_results.json    # Main results
│   │   ├── final_summary.csv              # Summary table
│   │   ├── component_analysis.json        # Component analysis
│   │   ├── publication_summary.md         # Paper-ready summary
│   │   └── tables/                        # LaTeX tables
│   ├── competitors/                        # Baseline implementations
│   │   ├── strgnn/                        # StrGNN implementation
│   │   └── bright/                        # BRIGHT implementation
│   └── run_tgb_expansion.slurm            # Main execution script
├── src/                                    # Core implementations
│   ├── models/                            # Model architectures
│   ├── data/                              # Data processing
│   └── training/                          # Training pipelines
├── assess_paper_results/                   # Verification pipeline
│   ├── final_verification_with_exact_methodology.py
│   └── fixed_ablation_study.py
├── temporal_anomaly_detector.py            # Main TempAnom-GNN
├── bitcoin_baseline_comparison.py          # Baseline methods
├── temporal_memory_module.py               # Memory components
├── static_gnn_baseline.py                 # Static baselines
└── set_temporal_gnn                       # Environment activation
```

## 🔬 Reproducing Key Results

### 1. Cross-Dataset Consistency Validation

```bash
# Run on both Bitcoin datasets
cd multi_dataset_expansion
python -c "
import pandas as pd
import json

# Load results
with open('results/final_expansion_results.json', 'r') as f:
    results = json.load(f)

# Extract separation ratios
alpha_sep = results['bitcoin_alpha']['negative_ratio']['separation_ratio']
otc_sep = results['bitcoin_otc']['negative_ratio']['separation_ratio']

print(f'Bitcoin Alpha: {alpha_sep:.2f}× separation')
print(f'Bitcoin OTC: {otc_sep:.2f}× separation')
print(f'Consistency: {abs(alpha_sep - otc_sep) / max(alpha_sep, otc_sep) * 100:.1f}% difference')
"

# Expected output:
# Bitcoin Alpha: 25.08× separation
# Bitcoin OTC: 25.97× separation  
# Consistency: 3.6% difference
```

### 2. Component Analysis (Evolution-Only Dominance)

```bash
# Run component analysis
python assess_paper_results/fixed_ablation_study.py

# View component results
python -c "
import json

with open('multi_dataset_expansion/results/component_analysis.json', 'r') as f:
    components = json.load(f)

for dataset, results in components.items():
    print(f'\n{dataset.upper()}:')
    sorted_components = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for i, (comp, metrics) in enumerate(sorted_components):
        marker = '🏆' if i == 0 else '  '
        print(f'{marker} {comp}: {metrics["auc"]:.3f} AUC')
"

# Expected: Evolution-only dominance across all datasets
```

### 3. Ground Truth Validation

```bash
# Verify exact ground truth methodology
python -c "
import pandas as pd
from collections import defaultdict

def validate_ground_truth(dataset_path, expected_count):
    df = pd.read_csv(dataset_path)
    
    # EXACT methodology from paper
    user_stats = defaultdict(lambda: {'total': 0, 'negative': 0})
    for _, row in df.iterrows():
        target = row['target_idx']
        user_stats[target]['total'] += 1
        if row['rating'] < 0:  # CRITICAL: < 0, not == -1
            user_stats[target]['negative'] += 1
    
    suspicious_users = set()
    for user, stats in user_stats.items():
        if stats['total'] >= 5:  # ≥5 interactions
            neg_ratio = stats['negative'] / stats['total']
            if neg_ratio > 0.3:  # >30% negative
                suspicious_users.add(user)
    
    print(f'{dataset_path}: {len(suspicious_users)} suspicious users (expected: {expected_count})')
    return len(suspicious_users) == expected_count

# Validate both datasets
alpha_valid = validate_ground_truth('data/processed/bitcoin_alpha_processed.csv', 73)
otc_valid = validate_ground_truth('data/processed/bitcoin_otc_processed.csv', 219)

print(f'\nValidation: {"✅ PASSED" if alpha_valid and otc_valid else "❌ FAILED"}')
"

# Expected output:
# Bitcoin Alpha: 73 suspicious users (expected: 73)
# Bitcoin OTC: 219 suspicious users (expected: 219)
# Validation: ✅ PASSED
```

## 🧪 Running Individual Components

### TempAnom-GNN Core Model

```bash
# Test temporal anomaly detector
python temporal_anomaly_detector.py

# Expected output:
# ✅ Temporal AUC: 0.750+
# ✅ Component analysis completed
# 📊 Results saved to results/
```

### Baseline Comparisons

```bash
# Run static baselines
python bitcoin_baseline_comparison.py

# Expected: Negative ratio achieving ~25× separation
```

### TGN Baseline

```bash
# Test TGN implementation
cd multi_dataset_expansion/competitors
python test_tgn_implementation.py

# Expected: TGN AUC around 0.80-0.82
```

## 📊 Performance Benchmarks

### Expected Runtimes (NVIDIA A100)

| Component | Time | Memory |
|-----------|------|--------|
| Full expansion | 5-15 min | 16GB |
| Single dataset | 2-5 min | 8GB |
| Baseline only | 1-2 min | 4GB |
| Component analysis | 3-5 min | 8GB |

### Performance Targets

| Metric | Bitcoin Alpha | Bitcoin OTC | TGB Wiki |
|--------|---------------|-------------|----------|
| TempAnom-GNN AUC | 0.75 ± 0.02 | 0.74 ± 0.02 | 0.73 ± 0.03 |
| Negative Ratio Sep | 25.08 ± 1.0 | 25.97 ± 1.0 | N/A |
| Evolution Dominance | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed |

## 🔧 Troubleshooting

### Common Issues

**Environment Issues:**
```bash
# CUDA not available
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.is_available())"

# PyG installation issues
pip uninstall torch-geometric
pip install torch-geometric==2.6.1
```

**Data Issues:**
```bash
# Missing datasets
python download_datasets.py

# Corrupted data
rm -rf data/processed/*
python download_datasets.py
```

**SLURM Issues:**
```bash
# Check queue
squeue -u $USER

# View logs
tail -f logs/tgb_expansion_*.out

# Run interactively if SLURM unavailable
cd multi_dataset_expansion
source ../set_temporal_gnn
bash run_tgb_expansion.slurm
```

### Performance Issues

```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use CPU if GPU unavailable
export CUDA_VISIBLE_DEVICES=""
```

## 📈 Extending the Work

### Adding New Datasets

1. **Create dataset loader** in `src/data/`
2. **Implement ground truth** methodology
3. **Add to expansion pipeline** in `multi_dataset_expansion/`
4. **Update evaluation** metrics

### Adding New Methods

1. **Implement competitor** in `competitors/`
2. **Create wrapper** for unified interface
3. **Add to methods** configuration
4. **Run comparative** evaluation

### Custom Analysis

```bash
# Create custom analysis script
cp multi_dataset_expansion/run_tgb_expansion.slurm my_analysis.slurm

# Modify for your specific needs
# Submit job
sbatch my_analysis.slurm
```

## 📚 Citation

```bibtex
@inproceedings{tempanom-gnn-2025,
  title={TempAnom-GNN: Temporal Graph Neural Networks for Real-time Anomaly Detection},
  author={[]},
  booktitle={Proceedings of 2025},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bitcoin datasets**: Stanford SNAP
- **TGB framework**: Temporal Graph Benchmark
- **Baseline methods**: StrGNN, BRIGHT, TGN authors
- **Computing resources**: NJIT Wulver HPC Cluster

## 📞 Contact

- **Author**: []
- **Email**: []
- **Institution**: []
- **Paper**: [Link to paper when published]

---

**🚀 Ready to reproduce our results? Start with the Quick Start guide above!**
