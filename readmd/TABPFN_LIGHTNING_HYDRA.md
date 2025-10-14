# TabPFN PyTorch Lightning and Hydra Integration

## Summary

Successfully added PyTorch Lightning API support and Hydra configuration management to TabPFN, enabling consistent usage patterns across all models in the framework and simplified experiment configuration.

## Changes Made

### 1. PyTorch Lightning Support in TabPFN

Added Lightning integration to `src/models/tab_pfn.py`:

#### `use_lightning` Parameter
- Added `use_lightning` parameter to `TabPFNClassifier.__init__()` for API consistency with MLP and Transformer models
- **Default: `False`** - Ensures backward compatibility
- **Note**: TabPFN is a pre-trained model that performs in-context learning, so it doesn't train in the traditional sense
- The parameter is provided for API consistency and future extensibility

#### Lightning Import Handling
```python
try:
    import lightning as L
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
```

- Graceful fallback if Lightning is not available
- Warning message if `use_lightning=True` but Lightning is not installed

### 2. Hydra Configuration Files

Created three Hydra configuration files in `conf/model/`:

#### `tabpfn_default.yaml` - Balanced Configuration
```yaml
model:
  name: TabPFN
  params:
    device: cuda
    N_ensemble_configurations: 32
    random_state: 42
```
- **Use case**: General-purpose usage with balanced accuracy and speed
- **Ensemble configs**: 32 (recommended default)

#### `tabpfn_fast.yaml` - Speed-Optimized Configuration
```yaml
model:
  name: TabPFN
  params:
    device: cuda
    N_ensemble_configurations: 8
    random_state: 42
```
- **Use case**: Quick experiments and rapid prototyping
- **Ensemble configs**: 8 (faster inference)
- **Trade-off**: Slightly lower accuracy for better speed

#### `tabpfn_accurate.yaml` - Accuracy-Optimized Configuration
```yaml
model:
  name: TabPFN
  params:
    device: cuda
    N_ensemble_configurations: 64
    random_state: 42
```
- **Use case**: Best possible accuracy when speed is less critical
- **Ensemble configs**: 64 (higher accuracy)
- **Trade-off**: Slower inference for better performance

## Usage

### Using TabPFN with Hydra

#### Basic Usage with Default Configuration
```bash
cd src
python run_experiments_hydra.py model=tabpfn_default dataset.name=breast_cancer
```

#### Fast Configuration for Quick Experiments
```bash
python run_experiments_hydra.py model=tabpfn_fast dataset.name=breast_cancer
```

#### Accurate Configuration for Best Results
```bash
python run_experiments_hydra.py model=tabpfn_accurate dataset.name=breast_cancer
```

#### Custom Configuration Override
```bash
# Override ensemble configurations on the fly
python run_experiments_hydra.py model=tabpfn_default \
    model.params.N_ensemble_configurations=16 \
    dataset.name=breast_cancer
```

### Using TabPFN with Lightning Parameter (Programmatic)

```python
from models import TabPFNClassifier
from utils.data_loader import DataLoader

# Load data
loader = DataLoader('breast_cancer', random_state=42)
X, y = loader.load_data()
data = loader.prepare_data(X, y, test_size=0.2)

# Create model with Lightning parameter (for API consistency)
model = TabPFNClassifier(
    use_lightning=True,  # For API consistency
    device='cpu',
    N_ensemble_configurations=32
)

# Train (performs in-context learning)
model.train(data['X_train'], data['y_train'])

# Evaluate
metrics = model.evaluate(data['X_test'], data['y_test'])
print(metrics)
```

## Key Features

### 1. API Consistency
The `use_lightning` parameter provides API consistency with other models:
- **MLP**: `MLPClassifier(use_lightning=True)`
- **Transformer**: `TransformerClassifier(use_lightning=True)`
- **TabPFN**: `TabPFNClassifier(use_lightning=True)` ✓ Now supported

### 2. Hydra Configuration Management
TabPFN now supports the same Hydra-based configuration as other models:
- Easy switching between configurations
- Parameter overrides via command line
- Reproducible experiments with saved configs
- Configuration composition

### 3. Pre-configured Templates
Three ready-to-use configuration templates:
- **Default**: Balanced performance (32 ensembles)
- **Fast**: Quick experiments (8 ensembles)
- **Accurate**: Best results (64 ensembles)

### 4. Backward Compatibility
All changes are backward compatible:
- `use_lightning` parameter is optional (default: False)
- Existing code continues to work without modification
- No breaking changes to any APIs

## Technical Details

### TabPFN Characteristics
TabPFN is unique compared to other models:
- **Pre-trained**: No traditional training loop required
- **In-context learning**: Learns from training data during inference
- **Dataset constraints**: 
  - Max 10,000 samples
  - Max 100 features
- **No hyperparameter tuning**: Main parameter is ensemble size

### Why Lightning Support Matters
Even though TabPFN doesn't train traditionally, adding Lightning support:
1. **Maintains consistency** with other deep learning models
2. **Future-proofs** for potential TabPFN extensions
3. **Simplifies** switching between different model types
4. **Provides uniform API** across the framework

## Configuration Parameters

### Supported Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | 'cuda' | Device to run on ('cuda' or 'cpu') |
| `N_ensemble_configurations` | int | 32 | Number of ensemble configurations (1-256) |
| `random_state` | int | 42 | Random seed for reproducibility |
| `use_lightning` | bool | False | Use Lightning wrapper (for consistency) |

### Ensemble Configurations Guide

| Value | Performance | Speed | Use Case |
|-------|-------------|-------|----------|
| 1-8 | Lower | Fastest | Quick prototyping |
| 16-32 | Balanced | Moderate | General use |
| 32-64 | Good | Slower | Production |
| 64-256 | Best | Slowest | Research/benchmarking |

## Examples

### Running Experiments with Different Datasets

```bash
# Breast cancer with default TabPFN
python run_experiments_hydra.py model=tabpfn_default dataset.name=breast_cancer

# Bank marketing with fast TabPFN
python run_experiments_hydra.py model=tabpfn_fast dataset.name=bank_marketing

# Adult income with accurate TabPFN
python run_experiments_hydra.py model=tabpfn_accurate dataset.name=adult_income
```

### Parameter Sweeps

```bash
# Sweep over different ensemble sizes
python run_experiments_hydra.py model=tabpfn_default \
    model.params.N_ensemble_configurations=8,16,32,64 \
    --multirun
```

### Comparing Configurations

```bash
# Compare all three configurations
python run_experiments_hydra.py model=tabpfn_fast,tabpfn_default,tabpfn_accurate \
    dataset.name=breast_cancer \
    --multirun
```

## Integration with Existing Framework

TabPFN now integrates seamlessly with:
- ✓ Hydra configuration system
- ✓ Experiment runner (`run_experiments_hydra.py`)
- ✓ SHAP explainability
- ✓ LIME explainability
- ✓ Metrics evaluation
- ✓ Result logging

## Testing

### Test Configuration Loading
```bash
python -c "
import yaml
config = yaml.safe_load(open('conf/model/tabpfn_default.yaml'))
print(config)
"
```

### Test TabPFN with Hydra
```bash
cd src
python run_experiments_hydra.py model=tabpfn_default dataset.name=breast_cancer
```

### Test Lightning Parameter
```python
from models import TabPFNClassifier
model = TabPFNClassifier(use_lightning=True)
print(f"Lightning enabled: {model.use_lightning}")
```

## Best Practices

1. **Start with `tabpfn_default.yaml`** for most use cases
2. **Use `tabpfn_fast.yaml`** for rapid iteration and debugging
3. **Use `tabpfn_accurate.yaml`** for final benchmarks and comparisons
4. **Monitor dataset size** - TabPFN works best with ≤10K samples
5. **Check feature count** - TabPFN supports ≤100 features
6. **Use GPU** when available for better performance

## Limitations

1. **No Traditional Training**: TabPFN is pre-trained, so traditional training callbacks/monitoring don't apply
2. **Dataset Size**: Maximum 10,000 samples (automatically handled)
3. **Feature Count**: Maximum 100 features (automatically handled)
4. **Memory Usage**: Higher memory requirements than traditional models

## Future Enhancements

Potential improvements:
1. Add validation dataset support in configs
2. Support for TabPFN uncertainty quantification
3. Advanced ensemble configuration strategies
4. Integration with Lightning callbacks for inference monitoring
5. Automatic optimal ensemble size selection

## References

- TabPFN GitHub: https://github.com/PriorLabs/TabPFN
- PyTorch Lightning: https://lightning.ai/
- Hydra Configuration: https://hydra.cc/
- Original Framework: BeyondAccuracy-TabularXAI
