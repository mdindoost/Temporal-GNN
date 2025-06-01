
# TempAnom-GNN Paper Verification Report
Generated on: 2025-06-01 12:12:00

## Summary
- **Total Tests Run**: 29
- **Tests Passed**: 24
- **Tests Failed**: 2
- **Warnings**: 3
- **Success Rate**: 82.8%

## Verification Categories

### Dataset Verification
- Bitcoin Alpha edges: 24186
- Bitcoin Alpha users: 3783
- Suspicious users identified: 73
- Ground truth threshold: 30.0%

### Experimental Results Verification
- Early detection improvement: 20.8%
- Cold start improvement: 13.2%
- Bootstrap samples: 30

### Ablation Study Verification
- Evolution-only early detection: 0.36
- Memory-only cold start: 0.493
- Component interference confirmed: 8.3% degradation

### Hyperparameter Verification
- All hyperparameters checked against paper claims
- Implementation details verified

## Files Generated
- `comprehensive_verification_report.json`: Complete verification data
- `data_verification/`: Dataset verification results
- `experiments/`: Re-run experiment results
- `figures/`: Regenerated paper figures
- `tables/`: Regenerated paper tables
- `statistical_tests/`: Statistical validation results

## Recommendations

❌ **2 critical errors found. Review required.**
- Dataset - Bitcoin Alpha File: File not found: data/bitcoin/soc-sign-bitcoin-alpha.csv
- Conclusions - Evolution-only outperforms full system by 25.8%: Values don't match within tolerance 1.0

⚠️ **3 warnings issued:**
- Hyperparameters - lambda_1: Parameter not found in actual implementation
- Hyperparameters - lambda_2: Parameter not found in actual implementation
- Hyperparameters - lambda_3: Parameter not found in actual implementation
