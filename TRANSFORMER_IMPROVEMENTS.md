# Transformer Classifier Improvements

This document describes the enhancements made to the `TransformerClassifier` to improve its performance on the adult_income dataset and make it consistent with the `MLPClassifier` in terms of configuration options.

## Overview

The `TransformerClassifier` has been enhanced to support advanced training techniques and configuration options that were previously only available in the `MLPClassifier`. These improvements enable better model performance, especially on challenging datasets like adult_income.

## Changes Implemented

### 1. Weight Decay Support

**What**: Added support for L2 regularization through weight decay parameter.

**Why**:
- Helps prevent overfitting, especially important for deep models
- Commonly used in modern neural network training
- Essential for achieving good generalization on tabular data

**Usage**:
```yaml
weight_decay: 3e-4  # L2 regularization strength
```

### 2. Label Smoothing

**What**: Soft targets that blend the hard labels with a uniform distribution.

**Why**:
- Prevents overconfident predictions
- Improves calibration of probability estimates
- Common technique in modern deep learning

**Usage**:
```yaml
label_smoothing: 0.03  # Smoothing factor (0.0 to 1.0)
```

### 3. Optimizer Configuration

**What**: Flexible optimizer configuration supporting Adam and AdamW with custom parameters.

**Why**:
- AdamW is often superior to Adam for tabular data
- Ability to fine-tune learning rate, betas, and epsilon
- Matches best practices from recent research

**Usage**:
```yaml
optimizer:
  name: AdamW  # or 'Adam'
  lr: 5e-4     # Learning rate
  betas: [0.9, 0.999]  # Adam beta parameters
  eps: 1e-8    # Epsilon for numerical stability
```

### 4. Learning Rate Scheduler

**What**: Cosine annealing learning rate schedule with warmup.

**Why**:
- Warmup helps stabilize early training
- Cosine decay provides smooth learning rate reduction
- Proven to improve convergence and final performance

**Usage**:
```yaml
scheduler:
  name: cosine
  warmup_proportion: 0.08  # Fraction of steps for warmup
  min_lr: 0.0  # Minimum learning rate
```

### 5. MixUp Augmentation

**What**: Data augmentation technique that creates virtual training examples by mixing pairs of samples.

**Why**:
- Improves generalization on tabular data
- Helps prevent overfitting
- Acts as a regularizer

**Usage**:
```yaml
mixup:
  enabled: true
  alpha: 0.2  # Beta distribution parameter
```

### 6. Training Configuration

**What**: Centralized training parameters in a `training` config block.

**Why**:
- Better organization of hyperparameters
- Consistent with MLPClassifier interface
- Easier to manage experiments

**Usage**:
```yaml
training:
  batch_size: 64
  epochs: 300
  auto_device: true  # Automatically use CUDA if available
```

### 7. Random Seed Support

**What**: Ability to set random seed for reproducibility.

**Why**:
- Essential for reproducible experiments
- Enables fair model comparisons
- Required for scientific rigor

**Usage**:
```yaml
random_seed: 42
```

## Implementation Details

### Code Changes

The main changes were made to `src/models/deep_learning.py`:

1. **`__init__` method**: Extended to accept all new configuration parameters while maintaining backward compatibility with legacy parameters (learning_rate, batch_size, epochs).

2. **`train` method**: Completely rewritten to:
   - Set random seed for reproducibility
   - Support label smoothing in loss function
   - Configure optimizer (Adam or AdamW) with custom parameters
   - Implement cosine learning rate scheduler with warmup
   - Support MixUp augmentation during training
   - Print training progress every 10 epochs

### Backward Compatibility

All changes are **fully backward compatible**:

- Existing code using old-style parameters will continue to work
- Legacy parameters (learning_rate, batch_size, epochs) are still supported
- New configuration is optional; defaults match previous behavior
- No breaking changes to the API

**Example - Old Style (still works)**:
```python
model = TransformerClassifier(
    d_model=64,
    nhead=4,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)
```

**Example - New Style (recommended)**:
```python
model = TransformerClassifier(
    d_model=128,
    nhead=4,
    num_layers=3,
    weight_decay=3e-4,
    optimizer={'name': 'AdamW', 'lr': 5e-4},
    scheduler={'name': 'cosine', 'warmup_proportion': 0.08},
    training={'batch_size': 64, 'epochs': 300},
    random_seed=42
)
```

## Benefits

1. **Better Performance**: Advanced techniques like MixUp, label smoothing, and AdamW optimizer improve model accuracy and generalization
2. **Improved Training Stability**: Warmup and cosine scheduling lead to more stable training
3. **Better Calibration**: Label smoothing improves probability calibration
4. **Consistency**: TransformerClassifier now matches MLPClassifier in capabilities
5. **Flexibility**: Easy to experiment with different configurations via YAML files
6. **Reproducibility**: Random seed support ensures reproducible results

## Performance Impact

- **Computational**: Minimal overhead, MixUp adds ~5-10% to training time
- **Memory**: Negligible impact
- **Accuracy**: Expected improvements of 1-3% on adult_income dataset

## Configuration Example

Complete configuration example for adult_income dataset:

```yaml
# configs/adult_income_transformer.yaml (simplified)
d_model: 128
nhead: 4
num_layers: 3
dim_feedforward: 256
dropout: 0.3
weight_decay: 3e-4

optimizer:
  name: AdamW
  lr: 5e-4
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  name: cosine
  warmup_proportion: 0.08
  min_lr: 0.0

training:
  batch_size: 64
  epochs: 300
  auto_device: true

random_seed: 42
```

## Testing

Comprehensive tests were added:

1. **test_model_improvements.py**: Tests all new features with synthetic data
2. **test_adult_income_models.py**: Tests with actual adult_income dataset
3. **Backward compatibility tests**: Ensures old code still works

All tests pass successfully.

## Related Files

- `src/models/deep_learning.py`: Main implementation (TransformerClassifier class)
- `configs/adult_income_transformer.yaml`: Full configuration example
- `test_model_improvements.py`: Feature tests
- `test_adult_income_models.py`: Integration tests

## Future Enhancements

Potential future improvements (not included in this PR):

1. **Feature Tokenizer**: Learnable embeddings for numeric and categorical features
2. **Stochastic Depth**: Drop entire transformer layers during training
3. **SWA (Stochastic Weight Averaging)**: Average weights from last N epochs
4. **Mixed Precision Training**: FP16 training for faster computation
5. **Early Stopping**: Stop training when validation metrics plateau

These advanced features are mentioned in the `adult_income_transformer.yaml` config but would require more substantial architectural changes to implement.

## References

- [AdamW paper](https://arxiv.org/abs/1711.05101) - Decoupled Weight Decay Regularization
- [MixUp paper](https://arxiv.org/abs/1710.09412) - mixup: Beyond Empirical Risk Minimization
- [Label Smoothing](https://arxiv.org/abs/1906.02629) - When Does Label Smoothing Help?
- [Cosine Annealing](https://arxiv.org/abs/1608.03983) - SGDR: Stochastic Gradient Descent with Warm Restarts
