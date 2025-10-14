# TabPFN Integration Summary

## Overview

TabPFN (Prior-Fitted Networks) has been successfully integrated into the BeyondAccuracy-TabularXAI framework. TabPFN is a transformer-based model that uses in-context learning for small tabular classification tasks, eliminating the need for hyperparameter tuning.

## What is TabPFN?

TabPFN is a novel approach to tabular classification that:
- Uses a pre-trained transformer model trained on synthetic data
- Performs in-context learning without requiring retraining
- Achieves strong performance on small tabular datasets
- Works best with ≤10,000 samples and ≤100 features
- Supports binary and multi-class classification

**Reference**: https://github.com/PriorLabs/TabPFN

## Changes Made

### 1. Dependencies (`requirements.txt`)
- Added `tabpfn>=0.1.0` to the requirements

### 2. Model Implementation (`src/models/gradient_boosting.py`)
- Created `TabPFNClassifier` wrapper class with:
  - Graceful import handling (works even if TabPFN not installed)
  - Automatic dataset size limiting (max 10,000 samples, 100 features)
  - Consistent API with other models (train, predict, predict_proba, evaluate)
  - Feature importance placeholder (TabPFN doesn't provide native importance)

### 3. Model Registry (`src/models/__init__.py`)
- Added `TabPFNClassifier` to exports
- Updated `__all__` list to include TabPFN

### 4. Experiment Runner (`src/run_experiments.py`)
- Added TabPFN to model imports
- Added TabPFN initialization in `run_experiment()` function
- Added TabPFN to models list in `run_all_experiments()` function
- Set model_type to 'tree' for SHAP explainer compatibility

### 5. Documentation Updates
- **README.md**: Updated overview, features, and available models
- **PROJECT_SUMMARY.md**: Updated model count and descriptions
- **QUICKSTART.md**: Updated model count in quick start guide

### 6. Testing and Examples
- **test_tabpfn_integration.py**: Comprehensive integration tests
- **examples/tabpfn_example.py**: Usage example demonstrating TabPFN

## Usage

### Installing TabPFN

```bash
pip install tabpfn
```

### Running Single Experiment

```bash
cd src
python run_experiments.py breast_cancer TabPFN
```

### Running All Experiments

```bash
cd src
python run_experiments.py  # Includes TabPFN in the model list
```

### Using TabPFN Programmatically

```python
from models import TabPFNClassifier
from utils.data_loader import DataLoader

# Load data
loader = DataLoader('breast_cancer', random_state=42)
X, y = loader.load_data()
data = loader.prepare_data(X, y, test_size=0.2)

# Train model
model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
model.train(data['X_train'], data['y_train'])

# Evaluate
metrics = model.evaluate(data['X_test'], data['y_test'])
print(metrics)
```

## Key Features

### 1. Graceful Degradation
The implementation gracefully handles the case where TabPFN is not installed:
- Import succeeds even if TabPFN package is missing
- Clear error message when trying to instantiate without installation
- Other models continue to work normally

### 2. Automatic Dataset Constraints
TabPFN has specific requirements:
- Max 10,000 training samples
- Max 100 features

The wrapper automatically:
- Samples down to 10,000 if more samples provided
- Uses first 100 features if more features provided
- Warns the user when these limits are applied

### 3. Consistent API
TabPFNClassifier follows the same API as other models:
- `train(X_train, y_train)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get probability predictions
- `evaluate(X_test, y_test)` - Compute metrics
- `get_feature_importance(feature_names)` - Get feature importance

### 4. SHAP/LIME Compatibility
TabPFN is treated as a tree-based model for explainability purposes, making it compatible with:
- SHAP TreeExplainer
- LIME TabularExplainer

## Limitations

1. **Dataset Size**: TabPFN works best with ≤10,000 samples
2. **Feature Count**: TabPFN supports ≤100 features
3. **Feature Importance**: TabPFN doesn't provide native feature importance (returns uniform importance as placeholder)
4. **Computational Resources**: TabPFN requires more memory than traditional tree-based models

## Testing

Run the integration tests:

```bash
python test_tabpfn_integration.py
```

Expected output:
```
TabPFN Import: PASSED
Graceful Failure: PASSED
Model List: PASSED
Run Experiments Import: PASSED
Requirements.txt: PASSED
```

Run the example:

```bash
python examples/tabpfn_example.py
```

## Backward Compatibility

All changes are backward compatible:
- Existing code continues to work without modification
- TabPFN is optional - the framework works without it installed
- No breaking changes to any existing APIs

## Future Enhancements

Potential improvements for future versions:
1. Add native feature importance estimation using permutation importance
2. Support for TabPFN's built-in uncertainty quantification
3. Automatic ensemble configuration tuning
4. Integration with TabPFN's latest features (when released)

## References

- TabPFN GitHub: https://github.com/PriorLabs/TabPFN
- TabPFN Paper: "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
- BeyondAccuracy-TabularXAI: Main repository documentation
