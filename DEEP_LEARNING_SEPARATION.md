# Deep Learning Model Separation

## Overview

The deep learning models have been separated into **4 distinct variants** to provide clearer separation of concerns and make it explicit whether knowledge distillation is being used.

## The 4 Model Types

### 1. MLPClassifier
**Purpose**: Standard Multi-Layer Perceptron without distillation

**Usage**:
```python
from models import MLPClassifier

model = MLPClassifier(
    hidden_dims=[128, 64, 32],
    dropout=0.2,
    training={'batch_size': 128, 'epochs': 100}
)
model.train(X_train, y_train)
```

**Hydra Config**: `mlp_default.yaml`, `mlp_small.yaml`, `mlp_large.yaml`

### 2. TransformerClassifier
**Purpose**: Standard Transformer-based model without distillation

**Usage**:
```python
from models import TransformerClassifier

model = TransformerClassifier(
    d_model=64,
    nhead=4,
    num_layers=2,
    training={'batch_size': 128, 'epochs': 100}
)
model.train(X_train, y_train)
```

**Hydra Config**: `transformer_default.yaml`, `transformer_small.yaml`, `transformer_large.yaml`

### 3. MLPDistillationClassifier
**Purpose**: MLP with knowledge distillation enabled by default

**Usage**:
```python
from models import MLPDistillationClassifier, XGBoostClassifier

# Train teacher
teacher = XGBoostClassifier(n_estimators=100)
teacher.train(X_train, y_train)
teacher_probs = teacher.predict_proba(X_train)
teacher_logits = np.log(teacher_probs + 1e-10)

# Train distilled student
model = MLPDistillationClassifier(
    hidden_dims=[128, 64, 32],
    distillation={'lambda': 0.7, 'temperature': 2.0}
)
model.train(X_train, y_train, teacher_probs=teacher_logits)
```

**Hydra Config**: `mlp_distillation.yaml`

**Key Features**:
- Distillation is **enabled by default**
- Default λ (lambda) = 0.7 (70% teacher, 30% ground truth)
- Default temperature = 2.0
- Inherits from `MLPClassifier`

### 4. TransformerDistillationClassifier
**Purpose**: Transformer with knowledge distillation enabled by default

**Usage**:
```python
from models import TransformerDistillationClassifier, XGBoostClassifier

# Train teacher
teacher = XGBoostClassifier(n_estimators=100)
teacher.train(X_train, y_train)
teacher_probs = teacher.predict_proba(X_train)
teacher_logits = np.log(teacher_probs + 1e-10)

# Train distilled student
model = TransformerDistillationClassifier(
    d_model=64,
    nhead=4,
    distillation={'lambda': 0.7, 'temperature': 2.0}
)
model.train(X_train, y_train, teacher_probs=teacher_logits)
```

**Hydra Config**: `transformer_distillation.yaml`

**Key Features**:
- Distillation is **enabled by default**
- Default λ (lambda) = 0.7 (70% teacher, 30% ground truth)
- Default temperature = 2.0
- Inherits from `TransformerClassifier`

## Design Rationale

### Why Separate Classes?

1. **Clarity**: Makes it explicit which model uses distillation
2. **Ease of Use**: Distillation models have sensible defaults
3. **Backward Compatibility**: Original `MLPClassifier` and `TransformerClassifier` remain unchanged
4. **Type Safety**: Different classes for different behaviors
5. **Configuration Management**: Clear Hydra configs for each variant

### Inheritance Hierarchy

```
MLPClassifier
  └── MLPDistillationClassifier

TransformerClassifier
  └── TransformerDistillationClassifier
```

Both distillation variants inherit from their base classes, so they support all the same features (optimizer settings, schedulers, SWA, etc.).

## Using with Hydra

### Run Standard Models

```bash
# MLP without distillation
python run_experiments_hydra.py model=mlp_default

# Transformer without distillation
python run_experiments_hydra.py model=transformer_default
```

### Run Distillation Models

```bash
# MLP with distillation
python run_experiments_hydra.py model=mlp_distillation

# Transformer with distillation
python run_experiments_hydra.py model=transformer_distillation
```

### Override Distillation Parameters

```bash
# Adjust distillation weight
python run_experiments_hydra.py model=mlp_distillation \
  model.params.distillation.lambda=0.5

# Adjust temperature
python run_experiments_hydra.py model=mlp_distillation \
  model.params.distillation.temperature=3.0

# Disable distillation (effectively making it like mlp_default)
python run_experiments_hydra.py model=mlp_distillation \
  model.params.distillation.enabled=false
```

## Implementation Details

### Code Location
- **Classes**: `src/models/deep_learning.py`
- **Exports**: `src/models/__init__.py`
- **Hydra Support**: `src/run_experiments_hydra.py`
- **Configs**: `conf/model/mlp_distillation.yaml`, `conf/model/transformer_distillation.yaml`
- **Tests**: `tests/test_four_dl_models.py`, `tests/test_four_dl_models_syntax.py`

### Key Differences from Base Classes

The distillation classes differ from their base classes in only one way:

```python
# Base classes: distillation disabled by default
self.distillation = distillation or {'enabled': False}

# Distillation classes: distillation enabled by default
if distillation is None:
    distillation = {
        'enabled': True,
        'lambda': 0.7,
        'temperature': 2.0
    }
```

Everything else is inherited from the base class.

## Testing

Run tests to verify the separation:

```bash
# Syntax and structure tests (no dependencies required)
python tests/test_four_dl_models_syntax.py

# Full integration tests (requires numpy, torch, etc.)
python tests/test_four_dl_models.py
```

## Migration Guide

### From Old Code

If you were using distillation with the old approach:

**Before**:
```python
model = MLPClassifier(
    hidden_dims=[128, 64, 32],
    distillation={'enabled': True, 'lambda': 0.7, 'temperature': 2.0}
)
```

**After** (recommended):
```python
model = MLPDistillationClassifier(
    hidden_dims=[128, 64, 32]
    # distillation is enabled by default with lambda=0.7, temperature=2.0
)
```

**Both approaches still work!** The new classes just make the intention clearer.

## Summary

| Model Type | Class Name | Distillation | Config Files |
|------------|-----------|--------------|--------------|
| MLP (base) | `MLPClassifier` | Disabled | `mlp_default.yaml`, `mlp_small.yaml`, `mlp_large.yaml` |
| Transformer (base) | `TransformerClassifier` | Disabled | `transformer_default.yaml`, `transformer_small.yaml`, `transformer_large.yaml` |
| MLP with KD | `MLPDistillationClassifier` | **Enabled** | `mlp_distillation.yaml` |
| Transformer with KD | `TransformerDistillationClassifier` | **Enabled** | `transformer_distillation.yaml` |

This separation makes it clear which models use knowledge distillation and provides better defaults for each use case.
