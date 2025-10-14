# XGBoost → Deep Learning Distillation Feature

This feature implements knowledge distillation from XGBoost (teacher) to deep learning models (student), providing a significant performance boost for deep learning models on tabular data.

## Overview

Knowledge distillation is a technique where a smaller/simpler "student" model learns from a larger/more complex "teacher" model. This implementation allows deep learning models (MLP or Transformer) to learn from XGBoost's soft probability predictions, transferring not just correct predictions but also the uncertainty and confidence patterns from the teacher.

## Key Features

### 1. Temperature-Scaled Knowledge Distillation

- **Combined Loss Function**: L = λ * KL(teacher || student) + (1-λ) * CE(y, student)
  - `λ`: Weight for distillation loss (default: 0.7)
  - `KL`: Kullback-Leibler divergence between teacher and student probabilities
  - `CE`: Cross-entropy loss with ground truth labels
  
- **Temperature Scaling**: Applies temperature `T` (default: 2.0) to soften probability distributions
  - Higher temperature creates "softer" distributions with more information
  - Temperature is applied to both teacher and student logits
  - KL loss is scaled by T² to maintain gradient magnitude

### 2. SHAP-Based Consistency Penalty (Optional)

- Identifies top-k most important features using SHAP values from teacher
- Adds Jacobian norm penalty to encourage sensitivity to important features
- Helps student model focus on the features that matter most
- Small penalty weight (e.g., 0.01) is typically sufficient

## Usage

### Basic Distillation

```python
from models import XGBoostClassifier, MLPClassifier
import numpy as np

# 1. Train XGBoost teacher
teacher = XGBoostClassifier(n_estimators=200, max_depth=6)
teacher.train(X_train, y_train)

# 2. Get soft probabilities
teacher_probs = teacher.predict_proba(X_train)
teacher_logits = np.log(teacher_probs + 1e-10)

# 3. Train distilled MLP student
student = MLPClassifier(
    hidden_dims=[128, 64, 32],
    dropout=0.2,
    training={'batch_size': 64, 'epochs': 50},
    distillation={
        'enabled': True,
        'lambda': 0.7,        # Weight for distillation loss
        'temperature': 2.0    # Temperature for soft targets
    }
)
student.train(X_train, y_train, teacher_probs=teacher_logits)
```

### With SHAP Consistency Penalty

```python
import shap

# 1. Train teacher and get SHAP importance
teacher = XGBoostClassifier(n_estimators=200)
teacher.train(X_train, y_train)

explainer = shap.TreeExplainer(teacher.model)
shap_values = explainer.shap_values(X_train)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_k_indices = np.argsort(mean_abs_shap)[-5:].tolist()

# 2. Get teacher probabilities
teacher_probs = teacher.predict_proba(X_train)
teacher_logits = np.log(teacher_probs + 1e-10)

# 3. Train with consistency penalty
student = MLPClassifier(
    hidden_dims=[128, 64, 32],
    training={'batch_size': 64, 'epochs': 50},
    distillation={
        'enabled': True,
        'lambda': 0.7,
        'temperature': 2.0,
        'consistency_penalty': {
            'enabled': True,
            'top_k_features': top_k_indices,  # Feature indices
            'weight': 0.01                     # Penalty weight
        }
    }
)
student.train(X_train, y_train, teacher_probs=teacher_logits)
```

## Configuration Parameters

### Distillation Config

```python
distillation = {
    'enabled': bool,              # Enable distillation (default: False)
    'lambda': float,              # Weight for distillation loss (default: 0.7)
    'temperature': float,         # Temperature for soft targets (default: 2.0)
    'consistency_penalty': {      # Optional consistency penalty
        'enabled': bool,          # Enable penalty (default: False)
        'top_k_features': list,   # Feature indices for penalty
        'weight': float           # Penalty weight (default: 0.01)
    }
}
```

### Recommended Values

- **λ (lambda)**: 0.7 (70% distillation, 30% hard labels)
  - Higher values (0.8-0.9) rely more on teacher
  - Lower values (0.5-0.6) rely more on ground truth
  
- **Temperature (T)**: 2.0
  - Range: 1.0-5.0
  - Higher values create softer distributions with more information
  - Lower values are closer to hard labels
  
- **Consistency Weight**: 0.01
  - Range: 0.001-0.1
  - Should be small to avoid overwhelming main loss

## Examples

### 1. Basic Distillation Example
```bash
python examples/xgboost_distillation_example.py
```

This example demonstrates:
- Training XGBoost teacher
- Training baseline MLP and Transformer
- Training distilled MLP and Transformer
- Performance comparison

### 2. SHAP Consistency Penalty Example
```bash
python examples/distillation_with_shap_penalty_example.py
```

This example demonstrates:
- Computing SHAP feature importance
- Identifying top-k important features
- Training with consistency penalty
- Comparing with and without penalty

## How It Works

### 1. Knowledge Distillation Loss

The distillation loss combines soft targets from the teacher with hard labels:

```
L_distill = λ * KL(P_teacher || P_student) + (1-λ) * CE(y, P_student)
```

Where:
- `P_teacher = softmax(logits_teacher / T)`: Teacher's soft targets
- `P_student = softmax(logits_student / T)`: Student's soft predictions
- `T`: Temperature parameter
- `y`: Ground truth labels

### 2. Temperature Scaling

Temperature makes probability distributions "softer":

```python
# Before temperature (T=1): [0.9, 0.1]  - hard distribution
# After temperature (T=2):  [0.7, 0.3]  - soft distribution
```

Soft distributions contain more information about the model's uncertainty and relationships between classes.

### 3. Consistency Penalty

The Jacobian norm penalty encourages sensitivity to important features:

```
L_penalty = -weight * mean(||∂output/∂x_k||²)
```

Where:
- `x_k`: Top-k features identified by SHAP
- Negative sign encourages larger gradients (more sensitivity)
- Applied only to important features

## Benefits

1. **Improved Performance**: Student model learns from teacher's knowledge
2. **Better Calibration**: Soft targets improve probability calibration
3. **Uncertainty Transfer**: Student learns teacher's uncertainty patterns
4. **Feature Focus**: Consistency penalty emphasizes important features
5. **Single Biggest Boost**: Often provides the largest performance gain from a single technique

## Supported Models

- **Teacher**: XGBoostClassifier (recommended), any model with `predict_proba()`
- **Student**: 
  - MLPClassifier
  - TransformerClassifier

## Limitations

1. Distillation and MixUp are mutually exclusive (distillation takes precedence)
2. Consistency penalty adds computational overhead during training
3. Requires teacher model to be trained first
4. Teacher probabilities must match student's training data size

## Testing

Run the distillation tests:

```bash
python test_distillation.py
```

This tests:
- Basic MLP distillation
- Basic Transformer distillation
- Different lambda and temperature values
- Backward compatibility

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network"
2. Temperature scaling for knowledge distillation
3. SHAP (SHapley Additive exPlanations) for feature importance

## Implementation Details

### File Changes

- `src/kd.py`:
  - Core knowledge distillation utilities module
  - `compute_distillation_loss()`: Computes combined KL divergence and cross-entropy loss
  - `compute_consistency_penalty()`: Computes Jacobian norm penalty for feature importance
  - `should_enable_distillation()`: Helper to check if distillation is enabled
  - `get_distillation_params()`: Extracts distillation parameters from configuration

- `src/models/deep_learning.py`:
  - Added `distillation` parameter to MLPClassifier and TransformerClassifier
  - Added `teacher_probs` parameter to `train()` methods
  - Uses `kd.py` module for distillation loss computation

### New Files

- `src/kd.py`: Knowledge distillation utilities module
- `examples/xgboost_distillation_example.py`: Basic distillation demo
- `examples/distillation_with_shap_penalty_example.py`: SHAP penalty demo
- `test_distillation.py`: Comprehensive test suite
- `DISTILLATION_FEATURE.md`: This documentation

## Support

For issues or questions:
1. Check the example scripts in `examples/`
2. Review this documentation
3. Run the test suite: `python test_distillation.py`
4. Open an issue on GitHub

## License

This feature follows the same license as the BeyondAccuracy-TabularXAI project.
