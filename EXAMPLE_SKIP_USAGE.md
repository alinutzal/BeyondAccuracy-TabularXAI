# Skip Duplicate Experiments - Usage Examples

This document demonstrates the new feature that prevents running duplicate experiments.

## Feature Overview

The experiment runner now automatically detects when results already exist for a dataset-model combination and skips the experiment to save time. This is particularly useful when:

- Running the full experiment suite multiple times
- Re-running experiments after a partial failure
- Iteratively adding new dataset-model combinations
- Working with long-running experiments

## How It Works

The system checks for the presence of `results.json` file in the experiment directory:
- **Path checked**: `results/{dataset_name}_{model_name}/results.json`
- **If exists**: Experiment is skipped and existing results are loaded
- **If not exists or --rerun flag**: Experiment runs normally

## Usage Examples

### Example 1: First Time Running Experiments

```bash
cd src
python run_experiments.py breast_cancer XGBoost
```

**Output:**
```
================================================================================
Running experiment: breast_cancer + XGBoost
================================================================================

Loading data...
Training XGBoost...
Evaluating model...
...
Results saved to results/breast_cancer_XGBoost
```

### Example 2: Running the Same Experiment Again (Automatic Skip)

```bash
cd src
python run_experiments.py breast_cancer XGBoost
```

**Output:**
```
================================================================================
Skipping experiment: breast_cancer + XGBoost
Results already exist at: results/breast_cancer_XGBoost
Use --rerun flag to force rerun this experiment
================================================================================
```

### Example 3: Force Rerun with --rerun Flag

```bash
cd src
python run_experiments.py breast_cancer XGBoost --rerun
```

**Output:**
```
================================================================================
Running experiment: breast_cancer + XGBoost
================================================================================

Loading data...
Training XGBoost...
...
```

### Example 4: Run All Experiments (Skips Existing)

```bash
cd src
python run_experiments.py
```

**Output:**
```
# Runs all 9 combinations (3 datasets × 3 models)
# Automatically skips any that already have results
# Only runs the missing ones

Skipping experiment: breast_cancer + XGBoost
Results already exist at: results/breast_cancer_XGBoost
...

Running experiment: adult_income + XGBoost
...

Running experiment: bank_marketing + LightGBM
...
```

### Example 5: Rerun All Experiments

```bash
cd src
python run_experiments.py --rerun
```

**Output:**
```
# Runs all 9 combinations, overwriting any existing results
Running experiment: breast_cancer + XGBoost
...
Running experiment: breast_cancer + LightGBM
...
```

## Benefits

1. **Time Savings**: Avoid re-running expensive experiments that already completed
2. **Robustness**: Resume after crashes without losing completed work
3. **Flexibility**: Easily add new experiments to an existing batch
4. **Control**: Use --rerun when you need to regenerate results

## Technical Details

### What Gets Checked

The system only checks for the existence of `results.json`. If this file exists, the experiment is considered complete.

### What Gets Loaded

When skipping, the entire `results.json` is loaded and returned, which includes:
- Dataset and model names
- Timestamp
- Training and test metrics
- Interpretability metrics

### What Gets Skipped

When an experiment is skipped:
- Data loading
- Model training
- Model evaluation
- SHAP/LIME explanations
- Metric calculations
- File generation

All the existing files (plots, CSVs, etc.) are preserved and reused.

## Tips

1. **Partial Reruns**: If you want to rerun just one experiment, use:
   ```bash
   python run_experiments.py dataset_name model_name --rerun
   ```

2. **Check What Exists**: Before running, check the results directory:
   ```bash
   ls -la results/
   ```

3. **Selective Cleanup**: To rerun specific experiments, delete their directories:
   ```bash
   rm -rf results/breast_cancer_XGBoost
   python run_experiments.py breast_cancer XGBoost
   ```

4. **Batch Processing**: Run experiments in stages without fear of duplication:
   ```bash
   # Stage 1: Quick models
   python run_experiments.py breast_cancer XGBoost
   python run_experiments.py breast_cancer LightGBM
   
   # Stage 2: Later, add slow models (won't rerun stages 1)
   python run_experiments.py breast_cancer Transformer
   ```

## Troubleshooting

**Q: I updated my code but experiments are being skipped**  
A: Use `--rerun` flag to force regeneration with the new code.

**Q: My results.json is corrupted but experiments are being skipped**  
A: Delete the corrupted results directory and run again:
```bash
rm -rf results/dataset_model
python run_experiments.py dataset model
```

**Q: How do I know if an experiment was skipped?**  
A: The output will clearly show "Skipping experiment:" message.

**Q: Can I use --rerun for just part of my experiments?**  
A: Yes! Use it with specific dataset/model arguments:
```bash
python run_experiments.py breast_cancer XGBoost --rerun
```
