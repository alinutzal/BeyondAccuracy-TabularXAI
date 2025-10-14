# Hydra Quick Start Guide

This guide provides a quick overview of using Hydra for running experiments with different parameter configurations.

## What is Hydra?

Hydra enables you to:
- Switch between different model configurations without changing code
- Run parameter sweeps to find optimal hyperparameters
- Track and reproduce experiments with saved configurations

## Installation

Hydra is already included in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Run with Default Configuration

```bash
cd src
python run_experiments_hydra.py
```

This uses the default XGBoost configuration on the breast_cancer dataset.

### 2. Choose Different Models

**XGBoost:**
```bash
python run_experiments_hydra.py model=xgboost_default    # Balanced
python run_experiments_hydra.py model=xgboost_shallow    # Interpretable
python run_experiments_hydra.py model=xgboost_deep       # Accurate
python run_experiments_hydra.py model=xgboost_regularized # Robust
```

**LightGBM:**
```bash
python run_experiments_hydra.py model=lightgbm_default
python run_experiments_hydra.py model=lightgbm_shallow
python run_experiments_hydra.py model=lightgbm_deep
python run_experiments_hydra.py model=lightgbm_regularized
```

**GradientBoosting (sklearn):**
```bash
python run_experiments_hydra.py model=gradient_boosting_default
python run_experiments_hydra.py model=gradient_boosting_shallow
python run_experiments_hydra.py model=gradient_boosting_deep
```

**MLP (Deep Learning):**
```bash
python run_experiments_hydra.py model=mlp_default
python run_experiments_hydra.py model=mlp_small
python run_experiments_hydra.py model=mlp_large
```

**Transformer (Deep Learning):**
```bash
python run_experiments_hydra.py model=transformer_default
python run_experiments_hydra.py model=transformer_small
python run_experiments_hydra.py model=transformer_large
```

**TabPFN (Pre-trained Transformer):**
```bash
python run_experiments_hydra.py model=tabpfn_default    # Balanced (32 ensembles)
python run_experiments_hydra.py model=tabpfn_fast       # Fast (8 ensembles)
python run_experiments_hydra.py model=tabpfn_accurate   # Accurate (64 ensembles)
```

### 3. Change Dataset

```bash
python run_experiments_hydra.py dataset.name=breast_cancer
python run_experiments_hydra.py dataset.name=adult_income
python run_experiments_hydra.py dataset.name=bank_marketing
```

### 4. Override Parameters

```bash
# Change learning rate
python run_experiments_hydra.py model.params.learning_rate=0.01

# Change multiple parameters
python run_experiments_hydra.py \
  model.params.n_estimators=200 \
  model.params.max_depth=8
```

### 5. Combine Options

```bash
python run_experiments_hydra.py \
  model=xgboost_shallow \
  dataset.name=adult_income \
  model.params.n_estimators=300
```

## Parameter Sweeps

Run multiple experiments with different parameter values:

```bash
# Sweep learning rate
python run_experiments_hydra.py -m \
  model.params.learning_rate=0.01,0.05,0.1

# Grid search
python run_experiments_hydra.py -m \
  model.params.n_estimators=100,200 \
  model.params.max_depth=3,6,9
```

## Quick Comparison

### Original Method (still works)
```bash
python run_experiments.py breast_cancer XGBoost
```

### Hydra Method
```bash
python run_experiments_hydra.py \
  model=xgboost_default \
  dataset.name=breast_cancer
```

## Available Configurations

### XGBoost Variants

| Configuration | Description | Key Settings |
|--------------|-------------|--------------|
| `xgboost_default` | Balanced performance | depth=6, lr=0.1 |
| `xgboost_shallow` | Better interpretability | depth=3, lr=0.05, more trees |
| `xgboost_deep` | Better accuracy | depth=10, lr=0.03 |
| `xgboost_regularized` | Prevents overfitting | L1/L2 regularization |

### LightGBM Variants

| Configuration | Description | Key Settings |
|--------------|-------------|--------------|
| `lightgbm_default` | Balanced performance | depth=6, lr=0.1 |
| `lightgbm_shallow` | Better interpretability | depth=3, lr=0.05, controlled leaves |
| `lightgbm_deep` | Better accuracy | depth=10, lr=0.03, many leaves |
| `lightgbm_regularized` | Prevents overfitting | L1/L2 regularization |

### GradientBoosting Variants

| Configuration | Description | Key Settings |
|--------------|-------------|--------------|
| `gradient_boosting_default` | Standard sklearn | depth=3, lr=0.1 |
| `gradient_boosting_shallow` | Very interpretable | depth=2, more trees |
| `gradient_boosting_deep` | More complex patterns | depth=5 |

## Output

Each experiment creates a directory with results:
```
results/{dataset}_{model}_{hash}/
├── hydra_config.yaml           # Configuration used
├── results.json                # Metrics
├── shap_summary.png            # SHAP plots
└── ...
```

## Common Use Cases

### 1. Find Best Learning Rate
```bash
python run_experiments_hydra.py -m \
  model=xgboost_default \
  model.params.learning_rate=0.001,0.01,0.05,0.1,0.2
```

### 2. Compare Shallow vs Deep Trees
```bash
python run_experiments_hydra.py model=xgboost_shallow
python run_experiments_hydra.py model=xgboost_deep
```

### 3. Test Across Datasets
```bash
python run_experiments_hydra.py -m \
  model=lightgbm_default \
  dataset.name=breast_cancer,adult_income,bank_marketing
```

### 4. Quick Hyperparameter Search
```bash
python run_experiments_hydra.py -m \
  model=xgboost_default \
  model.params.n_estimators=50,100,200 \
  model.params.max_depth=3,6,9
```

## Tips

1. **Start Simple**: Use default configurations first
2. **Small Changes**: Override one parameter at a time
3. **Save Configs**: Hydra automatically saves all configs
4. **Use Sweeps**: Multirun mode (-m) for parameter tuning

## Next Steps

- See [HYDRA_USAGE.md](HYDRA_USAGE.md) for detailed documentation
- Check [examples/hydra_example.py](examples/hydra_example.py) for code examples
- Read the [Hydra documentation](https://hydra.cc/) for advanced features

## Troubleshooting

**Issue: Cannot find config file**
- Make sure you're in the `src/` directory

**Issue: Parameter override not working**
- Use dot notation: `model.params.learning_rate=0.01`
- Check the config file structure

**Issue: Too many experiments with multirun**
- Reduce parameter combinations
- Run in batches

## Questions?

Open an issue on GitHub or check the main documentation.
