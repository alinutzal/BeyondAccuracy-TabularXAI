# Deep Learning Models with Hydra

This guide shows how to use Hydra configuration management for deep learning experiments (MLP and Transformer models).

## Model Variants

The repository now provides **4 distinct deep learning model types**:

1. **MLP** - Multi-Layer Perceptron without distillation
2. **Transformer** - Transformer-based model without distillation
3. **MLP_Distillation** - MLP with knowledge distillation from a teacher model
4. **Transformer_Distillation** - Transformer with knowledge distillation from a teacher model

## Available Configurations

### MLP (Multi-Layer Perceptron)

**Default Configuration** (`mlp_default.yaml`)
- Hidden layers: [128, 64, 32]
- Activation: ReLU
- Batch size: 128
- Epochs: 100
- Distillation: Disabled
- Best for: General-purpose experiments

**Small Configuration** (`mlp_small.yaml`)
- Hidden layers: [64, 32]
- Activation: SiLU
- Batch size: 128
- Epochs: 200
- Features: Gaussian noise, cosine scheduler, SWA
- Distillation: Disabled
- Best for: Small datasets (< 1000 samples)

**Large Configuration** (`mlp_large.yaml`)
- Hidden layers: [512, 512, 256, 256]
- Activation: GEGLU
- Batch size: 1024
- Epochs: 200
- Features: MixUp augmentation, label smoothing
- Distillation: Disabled
- Best for: Large datasets (> 10,000 samples)

**Distillation Configuration** (`mlp_distillation.yaml`) **[NEW]**
- Hidden layers: [128, 64, 32]
- Activation: ReLU
- Batch size: 128
- Epochs: 100
- **Distillation: Enabled (λ=0.7, temperature=2.0)**
- Best for: Learning from a teacher model (e.g., XGBoost)

### Transformer

**Default Configuration** (`transformer_default.yaml`)
- d_model: 64, heads: 4, layers: 2
- Batch size: 128
- Epochs: 100
- Distillation: Disabled
- Best for: General-purpose experiments

**Small Configuration** (`transformer_small.yaml`)
- d_model: 128, heads: 4, layers: 2
- Batch size: 128
- Epochs: 300
- Features: Gaussian noise, cosine scheduler, SWA
- Distillation: Disabled
- Best for: Small datasets

**Large Configuration** (`transformer_large.yaml`)
- d_model: 256, heads: 8, layers: 4
- Batch size: 1024
- Epochs: 300
- Features: MixUp augmentation, SWA
- Distillation: Disabled
- Best for: Large datasets

**Distillation Configuration** (`transformer_distillation.yaml`) **[NEW]**
- d_model: 64, heads: 4, layers: 2
- Batch size: 128
- Epochs: 100
- **Distillation: Enabled (λ=0.7, temperature=2.0)**
- Best for: Learning from a teacher model (e.g., XGBoost)

## Usage Examples

### Basic Usage

Run with default MLP configuration:
```bash
cd src
python run_experiments_hydra.py model=mlp_default
```

Run with default Transformer configuration:
```bash
python run_experiments_hydra.py model=transformer_default
```

### Choose Different Variants

```bash
# Small MLP for small datasets
python run_experiments_hydra.py model=mlp_small

# Large MLP for large datasets
python run_experiments_hydra.py model=mlp_large

# Small Transformer
python run_experiments_hydra.py model=transformer_small

# Large Transformer
python run_experiments_hydra.py model=transformer_large

# MLP with knowledge distillation (NEW)
python run_experiments_hydra.py model=mlp_distillation

# Transformer with knowledge distillation (NEW)
python run_experiments_hydra.py model=transformer_distillation
```

### Change Dataset

```bash
python run_experiments_hydra.py model=mlp_default dataset.name=adult_income
python run_experiments_hydra.py model=transformer_default dataset.name=breast_cancer
```

### Override Parameters

Change learning rate:
```bash
python run_experiments_hydra.py model=mlp_default model.params.optimizer.lr=0.01
```

Change batch size and epochs:
```bash
python run_experiments_hydra.py model=mlp_default \
  model.params.training.batch_size=256 \
  model.params.training.epochs=50
```

Change hidden dimensions for MLP:
```bash
python run_experiments_hydra.py model=mlp_default \
  model.params.hidden_dims=[256,128,64]
```

Change Transformer architecture:
```bash
python run_experiments_hydra.py model=transformer_default \
  model.params.d_model=128 \
  model.params.nhead=8 \
  model.params.num_layers=3
```

### Parameter Sweeps

Try different learning rates:
```bash
python run_experiments_hydra.py -m model=mlp_default \
  model.params.optimizer.lr=0.0001,0.001,0.01
```

Compare MLP architectures:
```bash
python run_experiments_hydra.py -m \
  model=mlp_small,mlp_default,mlp_large \
  dataset.name=breast_cancer
```

Test different dropout rates:
```bash
python run_experiments_hydra.py -m model=mlp_default \
  model.params.dropout=0.1,0.2,0.3,0.4,0.5
```

## Common Configurations

### For Small Datasets (< 1000 samples)

```bash
python run_experiments_hydra.py model=mlp_small dataset.name=breast_cancer
```

### For Medium Datasets (1000-10000 samples)

```bash
python run_experiments_hydra.py model=mlp_default dataset.name=adult_income
```

### For Large Datasets (> 10000 samples)

```bash
python run_experiments_hydra.py model=mlp_large dataset.name=bank_marketing
```

## Advanced Features

### Enable Data Augmentation

Enable MixUp:
```bash
python run_experiments_hydra.py model=mlp_default \
  model.params.mixup.enabled=true \
  model.params.mixup.alpha=0.2
```

Enable Gaussian noise:
```bash
python run_experiments_hydra.py model=mlp_default \
  model.params.gaussian_noise.enabled=true \
  model.params.gaussian_noise.std=0.01
```

### Enable Advanced Training Techniques

Enable SWA (Stochastic Weight Averaging):
```bash
python run_experiments_hydra.py model=mlp_default \
  model.params.swa.enabled=true \
  model.params.swa.final_epochs=20
```

Enable cosine annealing scheduler:
```bash
python run_experiments_hydra.py model=mlp_default \
  model.params.scheduler.name=cosine \
  model.params.scheduler.warmup_proportion=0.1 \
  model.params.scheduler.min_lr=0.0
```

### Combine Multiple Changes

```bash
python run_experiments_hydra.py model=mlp_default \
  dataset.name=adult_income \
  model.params.hidden_dims=[256,128,64] \
  model.params.optimizer.lr=0.005 \
  model.params.training.batch_size=512 \
  model.params.training.epochs=150 \
  model.params.mixup.enabled=true
```

## Knowledge Distillation

Knowledge distillation allows neural networks to learn from a "teacher" model (e.g., XGBoost) by using its soft predictions as additional training signal.

### Using Distillation Models

```bash
# Train with MLP distillation (requires teacher model predictions)
python run_experiments_hydra.py model=mlp_distillation

# Train with Transformer distillation
python run_experiments_hydra.py model=transformer_distillation
```

### Customizing Distillation Parameters

```bash
# Adjust distillation weight (lambda)
python run_experiments_hydra.py model=mlp_distillation \
  model.params.distillation.lambda=0.5  # 50% teacher, 50% ground truth

# Adjust temperature
python run_experiments_hydra.py model=mlp_distillation \
  model.params.distillation.temperature=3.0  # Softer predictions
```

### When to Use Distillation

- **Use `mlp_distillation` or `transformer_distillation` when**:
  - You have a well-trained teacher model (e.g., XGBoost)
  - You want to compress knowledge into a neural network
  - You need faster inference than the teacher model
  
- **Use regular `mlp_default` or `transformer_default` when**:
  - Training from scratch without a teacher
  - You don't have teacher model predictions

## Configuration Structure

All deep learning model configurations follow this structure:

```yaml
model:
  name: MLP  # or Transformer, MLP_Distillation, Transformer_Distillation
  params:
    # Architecture parameters
    hidden_dims: [128, 64, 32]  # for MLP
    # or
    d_model: 64                   # for Transformer
    nhead: 4
    num_layers: 2
    
    # Regularization
    dropout: 0.3
    weight_decay: 1e-4
    
    # Data augmentation
    mixup:
      enabled: false
      alpha: 0.0
    
    gaussian_noise:
      enabled: false
    
    # Knowledge distillation (for distillation models)
    distillation:
      enabled: true  # Only in *_distillation configs
      lambda: 0.7    # Weight for distillation loss
      temperature: 2.0  # Temperature for soft targets
    
    # Training parameters
    optimizer:
      name: Adam
      lr: 0.001
    
    training:
      batch_size: 128
      epochs: 100
      auto_device: true
    
    random_seed: 42
```

## Tips

1. **Start small**: Use `mlp_small` or `transformer_small` for initial experiments
2. **Use auto_device**: Set `training.auto_device=true` to automatically use GPU if available
3. **Early stopping**: Enable early stopping to prevent overfitting
4. **Hyperparameter search**: Use `-m` flag for multi-run parameter sweeps
5. **Save configurations**: Hydra automatically saves all configurations with results

## Results

Results are saved in the `results/` directory with:
- `results.json` - Performance metrics
- `hydra_config.yaml` - Complete configuration used
- SHAP visualizations
- Model checkpoints (if enabled)

## Comparison with Tree-Based Models

Run MLP and XGBoost on the same dataset:
```bash
python run_experiments_hydra.py model=mlp_default dataset.name=breast_cancer
python run_experiments_hydra.py model=xgboost_default dataset.name=breast_cancer
```

Compare all model types:
```bash
python run_experiments_hydra.py -m \
  model=mlp_default,transformer_default,xgboost_default,lightgbm_default \
  dataset.name=adult_income
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `model.params.training.batch_size=32`
- Use smaller model: `model=mlp_small`
- Reduce hidden dimensions

### Training Too Slow
- Increase batch size: `model.params.training.batch_size=512`
- Reduce epochs: `model.params.training.epochs=50`
- Use faster activation: Change from `geglu` to `relu`

### Underfitting
- Use larger model: `model=mlp_large`
- Increase epochs: `model.params.training.epochs=200`
- Reduce regularization: Lower dropout and weight decay

### Overfitting
- Use smaller model: `model=mlp_small`
- Increase regularization: Higher dropout and weight decay
- Enable data augmentation: MixUp, Gaussian noise
- Enable early stopping with appropriate patience

## Further Reading

- [Hydra Documentation](https://hydra.cc/)
- [Main Hydra Usage Guide](HYDRA_USAGE.md)
- [Quick Start Guide](HYDRA_QUICKSTART.md)
- [Get Started with Hydra](GET_STARTED_WITH_HYDRA.md)
