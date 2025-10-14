# Using Hydra for Configuration Management

This project now supports [Hydra](https://hydra.cc/) for flexible configuration management and parameter sweeps, particularly for gradient boosting models (XGBoost, LightGBM, and sklearn's GradientBoostingClassifier).

## Overview

Hydra enables you to:
- Easily switch between different parameter configurations
- Run parameter sweeps without modifying code
- Track and reproduce experiments with saved configurations
- Override any parameter from the command line

## Quick Start

### Basic Usage

Run an experiment with the default configuration:
```bash
cd src
python run_experiments_hydra.py
```

### Selecting Model Configurations

Choose from pre-defined model configurations:

**XGBoost variants:**
```bash
# Default configuration (balanced)
python run_experiments_hydra.py model=xgboost_default

# Shallow trees (better interpretability)
python run_experiments_hydra.py model=xgboost_shallow

# Deep trees (better accuracy)
python run_experiments_hydra.py model=xgboost_deep

# Regularized (prevents overfitting)
python run_experiments_hydra.py model=xgboost_regularized
```

**LightGBM variants:**
```bash
# Default configuration
python run_experiments_hydra.py model=lightgbm_default

# Shallow trees
python run_experiments_hydra.py model=lightgbm_shallow

# Deep trees
python run_experiments_hydra.py model=lightgbm_deep

# Regularized
python run_experiments_hydra.py model=lightgbm_regularized
```

**GradientBoosting (sklearn) variants:**
```bash
# Default configuration
python run_experiments_hydra.py model=gradient_boosting_default

# Shallow trees
python run_experiments_hydra.py model=gradient_boosting_shallow

# Deep trees
python run_experiments_hydra.py model=gradient_boosting_deep
```

### Changing Datasets

```bash
# Run on different datasets
python run_experiments_hydra.py dataset.name=breast_cancer
python run_experiments_hydra.py dataset.name=adult_income
python run_experiments_hydra.py dataset.name=bank_marketing
```

### Combining Options

```bash
# Combine model and dataset selection
python run_experiments_hydra.py model=xgboost_shallow dataset.name=adult_income

# Override specific parameters
python run_experiments_hydra.py model=xgboost_default model.params.n_estimators=200 model.params.max_depth=8
```

## Configuration Structure

### Directory Layout

```
conf/
├── config.yaml                      # Main configuration file
└── model/                           # Model configurations
    ├── xgboost_default.yaml
    ├── xgboost_shallow.yaml
    ├── xgboost_deep.yaml
    ├── xgboost_regularized.yaml
    ├── lightgbm_default.yaml
    ├── lightgbm_shallow.yaml
    ├── lightgbm_deep.yaml
    ├── lightgbm_regularized.yaml
    ├── gradient_boosting_default.yaml
    ├── gradient_boosting_shallow.yaml
    └── gradient_boosting_deep.yaml
```

### Main Configuration (config.yaml)

```yaml
defaults:
  - model: xgboost_default
  - _self_

dataset:
  name: breast_cancer
  test_size: 0.2
  random_state: 42
  scale_features: true

experiment:
  results_dir: ../results
  rerun: false

seed: 42
```

### Model Configuration Example (xgboost_shallow.yaml)

```yaml
# @package _global_
model:
  name: XGBoost
  params:
    n_estimators: 200
    max_depth: 3
    learning_rate: 0.05
    min_child_weight: 5
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
    eval_metric: logloss
```

## Advanced Usage

### Command-Line Parameter Overrides

Override any parameter directly from the command line:

```bash
# Override learning rate
python run_experiments_hydra.py model.params.learning_rate=0.01

# Override multiple parameters
python run_experiments_hydra.py \
  model.params.n_estimators=300 \
  model.params.max_depth=10 \
  model.params.learning_rate=0.05

# Change dataset split
python run_experiments_hydra.py dataset.test_size=0.3 dataset.random_state=123
```

### Parameter Sweeps (Multirun)

Run experiments with multiple parameter values:

```bash
# Sweep over learning rates
python run_experiments_hydra.py -m model.params.learning_rate=0.01,0.05,0.1

# Sweep over max_depth and learning_rate
python run_experiments_hydra.py -m \
  model.params.max_depth=3,6,10 \
  model.params.learning_rate=0.01,0.05,0.1

# This will run 9 experiments (3 depths × 3 learning rates)
```

### Custom Model Configurations

Create your own model configuration file in `conf/model/`:

**conf/model/xgboost_custom.yaml:**
```yaml
# @package _global_
model:
  name: XGBoost
  params:
    n_estimators: 500
    max_depth: 7
    learning_rate: 0.02
    min_child_weight: 3
    subsample: 0.85
    colsample_bytree: 0.85
    reg_alpha: 0.05
    reg_lambda: 0.5
    random_state: 42
    eval_metric: logloss
```

Then use it:
```bash
python run_experiments_hydra.py model=xgboost_custom
```

## Model Configuration Presets

### XGBoost Configurations

1. **xgboost_default**: Balanced performance
   - n_estimators: 100, max_depth: 6, learning_rate: 0.1

2. **xgboost_shallow**: Better interpretability
   - More trees, shallow depth (max_depth=3)
   - Lower learning rate (0.05)
   - Regularization via subsample and colsample

3. **xgboost_deep**: Better accuracy on complex patterns
   - Fewer trees, deeper depth (max_depth=10)
   - Very low learning rate (0.03)
   - Gamma regularization

4. **xgboost_regularized**: Prevents overfitting
   - Strong L1/L2 regularization
   - Moderate depth (max_depth=5)
   - Lower subsample ratios

### LightGBM Configurations

1. **lightgbm_default**: Balanced performance
   - n_estimators: 100, max_depth: 6, learning_rate: 0.1

2. **lightgbm_shallow**: Better interpretability
   - More trees with shallow depth
   - Controlled num_leaves (15)
   - Higher min_child_samples

3. **lightgbm_deep**: Better accuracy
   - Deeper trees with more leaves (127)
   - Lower min_child_samples for finer splits

4. **lightgbm_regularized**: Prevents overfitting
   - Strong L1/L2 regularization
   - Higher min_split_gain
   - Lower subsample ratios

### GradientBoosting (sklearn) Configurations

1. **gradient_boosting_default**: Standard sklearn GBM
   - n_estimators: 100, max_depth: 3, learning_rate: 0.1

2. **gradient_boosting_shallow**: Very shallow trees
   - max_depth: 2 for high interpretability
   - More trees to compensate

3. **gradient_boosting_deep**: Deeper trees
   - max_depth: 5 for complex patterns

## Results and Experiment Tracking

### Output Directory Structure

Each experiment creates a directory with a unique hash:
```
results/
└── {dataset}_{model}_{config_hash}/
    ├── hydra_config.yaml          # Full configuration used
    ├── results.json               # Performance metrics
    ├── shap_summary.png           # SHAP visualizations
    ├── shap_importance.png
    ├── shap_feature_importance.csv
    └── lime_importance.png
```

### Reproducing Experiments

To reproduce an experiment, use the saved configuration:
```bash
python run_experiments_hydra.py --config-path=../results/{experiment_dir} --config-name=hydra_config
```

## Backward Compatibility

The original `run_experiments.py` script continues to work with the existing YAML configs in the `configs/` directory. Hydra is an additional option, not a replacement.

**Original method (still works):**
```bash
python run_experiments.py breast_cancer XGBoost
```

**New Hydra method:**
```bash
python run_experiments_hydra.py model=xgboost_default dataset.name=breast_cancer
```

## Tips and Best Practices

1. **Start Simple**: Begin with default configurations before customizing
2. **Use Descriptive Names**: Create meaningful names for custom configurations
3. **Save Configurations**: Hydra automatically saves configs with results
4. **Parameter Sweeps**: Use multirun (-m) for systematic hyperparameter tuning
5. **Version Control**: Commit your custom configuration files to track experiments

## Examples

### Example 1: Compare XGBoost Variants on Same Dataset

```bash
cd src
python run_experiments_hydra.py model=xgboost_default dataset.name=breast_cancer
python run_experiments_hydra.py model=xgboost_shallow dataset.name=breast_cancer
python run_experiments_hydra.py model=xgboost_deep dataset.name=breast_cancer
python run_experiments_hydra.py model=xgboost_regularized dataset.name=breast_cancer
```

### Example 2: Test Model Across Datasets

```bash
python run_experiments_hydra.py -m \
  model=lightgbm_default \
  dataset.name=breast_cancer,adult_income,bank_marketing
```

### Example 3: Hyperparameter Grid Search

```bash
python run_experiments_hydra.py -m \
  model=xgboost_default \
  model.params.n_estimators=100,200,300 \
  model.params.max_depth=3,6,9 \
  model.params.learning_rate=0.01,0.05,0.1
```

This will run 27 experiments (3 × 3 × 3 combinations).

### Example 4: Create Custom Configuration for Specific Dataset

Create `conf/model/xgboost_adult_income.yaml`:
```yaml
# @package _global_
model:
  name: XGBoost
  params:
    n_estimators: 250
    max_depth: 8
    learning_rate: 0.03
    min_child_weight: 2
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
    eval_metric: logloss
```

Run it:
```bash
python run_experiments_hydra.py model=xgboost_adult_income dataset.name=adult_income
```

## Troubleshooting

### Issue: Config file not found
**Solution**: Make sure you're in the `src/` directory when running the script, or adjust the `config_path` in the decorator.

### Issue: Parameter override not working
**Solution**: Check the parameter path. Use dot notation: `model.params.learning_rate=0.01`

### Issue: Multirun creates too many experiments
**Solution**: Limit the parameter combinations or run them in batches.

## Further Reading

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Tutorials](https://hydra.cc/docs/tutorials/intro/)

## Questions or Issues?

If you encounter any problems or have questions about using Hydra with this project, please open an issue on GitHub.
