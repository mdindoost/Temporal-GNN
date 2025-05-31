# ABLATION STUDY EXPERIMENTAL CHECKLIST

## 🔬 EXPERIMENT SETUP

### Prerequisites
- [ ] Your temporal anomaly detector code is working
- [ ] You can modify component weights (α, β, γ) in your model
- [ ] You have evaluation functions for early detection and cold start
- [ ] Bitcoin Alpha dataset is loaded and ready

### Component Weight Configurations to Test:
1. [ ] Evolution Only: (1.0, 0.0, 0.0)
2. [ ] Memory Only: (0.0, 1.0, 0.0)  
3. [ ] Prediction Only: (0.0, 0.0, 1.0)
4. [ ] Equal Weights: (0.33, 0.33, 0.33)
5. [ ] Evolution Emphasis: (0.6, 0.3, 0.1)
6. [ ] Strong Evolution: (0.7, 0.2, 0.1)
7. [ ] Evolution + Memory: (0.5, 0.5, 0.0)
8. [ ] Evolution + Prediction: (0.5, 0.0, 0.5)

### Random Seeds to Use:
- [ ] Seed 42
- [ ] Seed 123
- [ ] Seed 456
- [ ] Seed 789
- [ ] Seed 999

## 📊 METRICS TO COLLECT

For each experiment, record:
- [ ] Configuration name
- [ ] Alpha, beta, gamma values
- [ ] Random seed used
- [ ] Early detection performance (Precision@10)
- [ ] Cold start performance (Precision@10)
- [ ] Training time (optional)
- [ ] Any notes about convergence

## 🔧 IMPLEMENTATION STEPS

### Step 1: Modify Your Model
```python
# Add this to your TemporalAnomalyDetector class
def set_component_weights(self, alpha, beta, gamma):
    self.alpha = alpha
    self.beta = beta  
    self.gamma = gamma
    
def compute_unified_anomaly_score(self, evolution_score, memory_score, trajectory_score):
    return (self.alpha * evolution_score + 
            self.beta * memory_score + 
            self.gamma * trajectory_score)
```

### Step 2: Create Experiment Loop
```python
# Use the template in run_ablation_experiments.py
# Replace placeholder functions with your actual code
```

### Step 3: Run Experiments
- [ ] Test one configuration first to make sure everything works
- [ ] Run all 8 configurations × 5 seeds = 40 experiments
- [ ] Save results after each experiment (in case of crashes)

### Step 4: Analyze Results
- [ ] Use analyze_results.py to process your data
- [ ] Generate LaTeX table for paper
- [ ] Create comparison visualizations

## 📋 EXPECTED OUTPUT

You should get a CSV file with columns:
- config_name, alpha, beta, gamma, seed, early_detection, cold_start

And a LaTeX table ready for your paper:
```
Configuration & Weights & Early Detection & Cold Start \\
Evolution Only & (1.0, 0.0, 0.0) & [YOUR RESULT] & [YOUR RESULT] \\
...
```

## ⏰ ESTIMATED TIME

- Setup: 30 minutes
- Running experiments: 2-4 hours (depending on your training time)
- Analysis: 30 minutes
- Total: 3-5 hours

## 🎯 SUCCESS CRITERIA

- [ ] All 40 experiments completed successfully
- [ ] Results show clear patterns (evolution-only should be best)
- [ ] Standard deviations are reasonable (not too high variance)
- [ ] LaTeX table is generated and ready for paper

## ❗ IMPORTANT NOTES

- Use the SAME evaluation setup as your main results
- Use the SAME train/test split for all experiments
- Record exact hyperparameters used
- Save intermediate results to avoid losing work
- Document any issues or anomalies in results
