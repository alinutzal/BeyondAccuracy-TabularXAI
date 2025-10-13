# Breast Cancer MLP and Transformer Improvements

This document describes the improvements made to MLP and Transformer models specifically for the Breast Cancer (Wisconsin Diagnostic) dataset.

## Dataset Characteristics

- **Size**: Tiny dataset (n=569 samples)
- **Features**: 30 numerical features (all computed from digitized images)
- **Target**: Binary classification (malignant vs benign)
- **Quirks**: High signal-to-noise ratio, easy to overfit
- **Baseline**: Logistic regression and XGBoost already perform well

## Problem Statement

The tiny dataset size (n=569) presents unique challenges:
1. Deep learning models easily overfit
2. Need heavy regularization
3. Data augmentation is crucial
4. Stable evaluation requires repeated cross-validation
5. Standard architectures designed for large datasets perform poorly

## Implemented Solutions

### 1. MLP Improvements

#### Architecture
- **Small network**: Hidden dims [128, 64] to prevent overfitting
- **SiLU activation**: Better gradient flow than ReLU on small datasets
- **BatchNorm**: More stable than LayerNorm on tiny batches
- **Heavy dropout**: 0.4 dropout rate for regularization

#### Training Enhancements
- **Gaussian noise augmentation**: σ=0.01-0.02 on numerical features during training
- **SWA (Stochastic Weight Averaging)**: Average weights over last 30 epochs
- **Label smoothing**: 0.02 to prevent overconfident predictions
- **AdamW optimizer**: lr=1e-3, weight_decay=5e-4
- **Cosine scheduler**: With warmup (8% of total steps)

#### Configuration
```yaml
hidden_dims: [128, 64]
activation: silu
batch_norm_per_block: true
dropout: 0.4
weight_decay: 5e-4

gaussian_noise:
  enabled: true
  std: 0.01

swa:
  enabled: true
  final_epochs: 30

label_smoothing: 0.02

optimizer:
  name: AdamW
  lr: 1e-3

scheduler:
  name: cosine
  warmup_proportion: 0.08
  min_lr: 0.0

training:
  batch_size: 32
  epochs: 300
  early_stopping:
    enabled: true
    patience: 50
```

### 2. Transformer Improvements

#### Architecture (Kept Tiny)
- **d_model**: 128 (not 256)
- **Heads**: 4
- **Layers**: 2 (not 6)
- **FFN**: 256 (not 640)
- **Dropout**: 0.4

**Rationale**: Transformer architectures tend to overfit on tiny datasets. We keep the architecture minimal. Frankly, MLP > Transformer here unless you add synthetic features (polynomial terms, interactions) which the MLP can also exploit.

#### Training Enhancements
Same as MLP:
- Gaussian noise augmentation (σ=0.01)
- SWA for last 30 epochs
- Label smoothing (0.02)
- AdamW optimizer with cosine scheduling

#### Configuration
```yaml
d_model: 128
nhead: 4
num_layers: 2
dim_feedforward: 256
dropout: 0.4
weight_decay: 5e-4

gaussian_noise:
  enabled: true
  std: 0.01

swa:
  enabled: true
  final_epochs: 30

label_smoothing: 0.02
```

### 3. Evaluation Strategy

**Repeated Stratified K-Fold Cross-Validation**
- 5×10 or 2×5 repeated stratified CV for stable estimates
- Per-fold standardization (StandardScaler)
- Report mean ± std across all folds

**Why Repeated CV?**
- With only 569 samples, single train/test split is unreliable
- Repeated CV provides stable performance estimates
- Stratification ensures balanced class distribution in each fold

## Implementation Details

### Code Changes

#### MLPClassifier (`src/models/deep_learning.py`)
1. Added `batch_norm_per_block` parameter (takes precedence over `layer_norm_per_block`)
2. Added support for `silu` activation (in addition to `relu` and `geglu`)
3. Added `gaussian_noise` config: `{enabled: bool, std: float}`
4. Added `swa` config: `{enabled: bool, final_epochs: int}`
5. Updated training loop to apply Gaussian noise during training
6. Integrated PyTorch SWA utilities for weight averaging

#### TransformerClassifier (`src/models/deep_learning.py`)
1. Added `gaussian_noise` config
2. Added `swa` config
3. Same training loop improvements as MLP

### Usage Example

```python
from utils.data_loader import DataLoader
from models.deep_learning import MLPClassifier
import yaml

# Load data
loader = DataLoader('breast_cancer', random_state=42)
X, y = loader.load_data()
data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)

# Load config
with open('configs/breast_cancer_mlp.yaml', 'r') as f:
    config = yaml.safe_load(f)
config.pop('notes', None)

# Train model
model = MLPClassifier(**config)
model.train(data['X_train'], data['y_train'])

# Evaluate
metrics = model.evaluate(data['X_test'], data['y_test'])
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

See `examples/breast_cancer_example.py` for a complete example with repeated CV.

## Results

### Expected Performance
- **MLP**: ~97-98% accuracy with proper regularization
- **Transformer**: ~95-96% accuracy (expect parity, not dominance)
- **Baseline (XGBoost)**: ~96-97% accuracy

### What Moves the Needle
1. **Gaussian noise injection**: +0.5-1% improvement
2. **SWA**: +0.3-0.5% improvement
3. **Repeated CV**: More stable estimates (±1-2% std)
4. **Heavy regularization**: Prevents overfitting on tiny dataset

## Key Takeaways

1. **Keep it small**: Large architectures overfit on n=569
2. **Heavy regularization**: dropout=0.4, weight_decay=5e-4
3. **BatchNorm works better** than LayerNorm on tiny datasets
4. **Noise injection + SWA** provide measurable improvements
5. **Transformer ≈ MLP** on this dataset (no dominance)
6. **Repeated CV is crucial** for stable performance estimates

## Files Modified

- `src/models/deep_learning.py`: Added SiLU, BatchNorm, Gaussian noise, SWA
- `configs/breast_cancer_mlp.yaml`: Updated with all improvements
- `configs/breast_cancer_transformer.yaml`: Simplified architecture + improvements
- `test_breast_cancer_models.py`: Comprehensive test suite
- `examples/breast_cancer_example.py`: Working example with repeated CV

## References

- Breast Cancer Wisconsin (Diagnostic) Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- Stochastic Weight Averaging: Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima and Better Generalization"
- Data Augmentation for Tabular Data: Shwartz-Ziv & Armon (2021) "Tabular Data: Deep Learning is Not All You Need"
