"""
Test script to verify breast cancer MLP and Transformer improvements.

Tests:
1. SiLU activation support
2. BatchNorm support (vs LayerNorm)
3. Gaussian noise augmentation
4. SWA (Stochastic Weight Averaging)
5. Configuration loading from YAML
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import yaml
from models.deep_learning import MLPClassifier, TransformerClassifier
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_data():
    """Load breast cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def test_mlp_silu_activation():
    """Test MLP with SiLU activation."""
    print("\n" + "="*80)
    print("Testing MLP with SiLU activation")
    print("="*80)
    
    X, y = load_breast_cancer_data()
    
    config = {
        'hidden_dims': [128, 64],
        'activation': 'silu',
        'dropout': 0.4,
        'training': {'batch_size': 32, 'epochs': 5}
    }
    
    model = MLPClassifier(**config)
    print(f"MLP initialized with activation: {model.activation}")
    assert model.activation == 'silu', "SiLU activation not set correctly"
    
    model.train(X, y)
    preds = model.predict(X[:10])
    
    print(f"Predictions: {preds[:5]}")
    print("✓ SiLU activation test passed!")
    return True

def test_mlp_batchnorm():
    """Test MLP with BatchNorm."""
    print("\n" + "="*80)
    print("Testing MLP with BatchNorm")
    print("="*80)
    
    X, y = load_breast_cancer_data()
    
    config = {
        'hidden_dims': [128, 64],
        'activation': 'silu',
        'batch_norm_per_block': True,
        'dropout': 0.4,
        'training': {'batch_size': 32, 'epochs': 5}
    }
    
    model = MLPClassifier(**config)
    print(f"MLP initialized with batch_norm_per_block: {model.batch_norm_per_block}")
    assert model.batch_norm_per_block == True, "BatchNorm not enabled"
    
    model.train(X, y)
    
    # Check that model contains BatchNorm layers
    has_batchnorm = any('BatchNorm' in str(type(m)) for m in model.model.modules())
    assert has_batchnorm, "Model does not contain BatchNorm layers"
    
    preds = model.predict(X[:10])
    print(f"Predictions: {preds[:5]}")
    print("✓ BatchNorm test passed!")
    return True

def test_mlp_gaussian_noise():
    """Test MLP with Gaussian noise augmentation."""
    print("\n" + "="*80)
    print("Testing MLP with Gaussian noise augmentation")
    print("="*80)
    
    X, y = load_breast_cancer_data()
    
    config = {
        'hidden_dims': [128, 64],
        'activation': 'silu',
        'batch_norm_per_block': True,
        'dropout': 0.4,
        'gaussian_noise': {'enabled': True, 'std': 0.01},
        'training': {'batch_size': 32, 'epochs': 5},
        'random_seed': 42
    }
    
    model = MLPClassifier(**config)
    print(f"MLP initialized with gaussian_noise: {model.gaussian_noise}")
    assert model.gaussian_noise.get('enabled') == True, "Gaussian noise not enabled"
    
    model.train(X, y)
    preds = model.predict(X[:10])
    
    print(f"Predictions: {preds[:5]}")
    print("✓ Gaussian noise augmentation test passed!")
    return True

def test_mlp_swa():
    """Test MLP with SWA (Stochastic Weight Averaging)."""
    print("\n" + "="*80)
    print("Testing MLP with SWA")
    print("="*80)
    
    X, y = load_breast_cancer_data()
    
    config = {
        'hidden_dims': [128, 64],
        'activation': 'silu',
        'batch_norm_per_block': True,
        'dropout': 0.4,
        'swa': {'enabled': True, 'final_epochs': 3},
        'training': {'batch_size': 32, 'epochs': 10},
        'random_seed': 42
    }
    
    model = MLPClassifier(**config)
    print(f"MLP initialized with swa: {model.swa_cfg}")
    assert model.swa_cfg.get('enabled') == True, "SWA not enabled"
    
    model.train(X, y)
    preds = model.predict(X[:10])
    
    print(f"Predictions: {preds[:5]}")
    print("✓ SWA test passed!")
    return True

def test_mlp_config_from_yaml():
    """Test loading MLP config from breast_cancer_mlp.yaml."""
    print("\n" + "="*80)
    print("Testing MLP with breast_cancer_mlp.yaml config")
    print("="*80)
    
    config_path = 'configs/breast_cancer_mlp.yaml'
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, skipping test")
        return True
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Remove non-model keys
    config.pop('notes', None)
    
    # Reduce epochs for testing
    if 'training' in config:
        config['training']['epochs'] = 10
    
    X, y = load_breast_cancer_data()
    
    model = MLPClassifier(**config)
    print(f"MLP loaded from config:")
    print(f"  Activation: {model.activation}")
    print(f"  BatchNorm: {model.batch_norm_per_block}")
    print(f"  Gaussian noise: {model.gaussian_noise}")
    print(f"  SWA: {model.swa_cfg}")
    print(f"  Label smoothing: {model.label_smoothing}")
    
    # Verify key settings
    assert model.activation == 'silu', f"Expected silu, got {model.activation}"
    assert model.batch_norm_per_block == True, "BatchNorm not enabled"
    assert model.gaussian_noise.get('enabled') == True, "Gaussian noise not enabled"
    assert model.swa_cfg.get('enabled') == True, "SWA not enabled"
    
    model.train(X, y)
    preds = model.predict(X[:10])
    
    print(f"Predictions: {preds[:5]}")
    print("✓ Config loading test passed!")
    return True

def test_transformer_gaussian_noise_and_swa():
    """Test Transformer with Gaussian noise and SWA."""
    print("\n" + "="*80)
    print("Testing Transformer with Gaussian noise and SWA")
    print("="*80)
    
    X, y = load_breast_cancer_data()
    
    config = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.4,
        'weight_decay': 5e-4,
        'gaussian_noise': {'enabled': True, 'std': 0.01},
        'swa': {'enabled': True, 'final_epochs': 3},
        'optimizer': {'name': 'AdamW', 'lr': 0.001},
        'training': {'batch_size': 32, 'epochs': 10, 'auto_device': True},
        'random_seed': 42
    }
    
    model = TransformerClassifier(**config)
    print(f"Transformer initialized with:")
    print(f"  d_model: {model.d_model}")
    print(f"  Gaussian noise: {model.gaussian_noise}")
    print(f"  SWA: {model.swa_cfg}")
    
    model.train(X, y)
    preds = model.predict(X[:10])
    
    print(f"Predictions: {preds[:5]}")
    print("✓ Transformer Gaussian noise and SWA test passed!")
    return True

def test_transformer_config_from_yaml():
    """Test loading Transformer config from breast_cancer_transformer.yaml."""
    print("\n" + "="*80)
    print("Testing Transformer with breast_cancer_transformer.yaml config")
    print("="*80)
    
    config_path = 'configs/breast_cancer_transformer.yaml'
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, skipping test")
        return True
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Remove non-model keys
    config.pop('notes', None)
    
    # Reduce epochs for testing
    if 'training' in config:
        config['training']['epochs'] = 10
    
    X, y = load_breast_cancer_data()
    
    model = TransformerClassifier(**config)
    print(f"Transformer loaded from config:")
    print(f"  d_model: {model.d_model}")
    print(f"  nhead: {model.nhead}")
    print(f"  num_layers: {model.num_layers}")
    print(f"  Gaussian noise: {model.gaussian_noise}")
    print(f"  SWA: {model.swa_cfg}")
    
    # Verify simplified config per issue requirements
    assert model.d_model == 128, f"Expected d_model=128, got {model.d_model}"
    assert model.nhead == 4, f"Expected nhead=4, got {model.nhead}"
    assert model.num_layers == 2, f"Expected num_layers=2, got {model.num_layers}"
    assert model.dim_feedforward == 256, f"Expected dim_feedforward=256, got {model.dim_feedforward}"
    assert model.dropout == 0.4, f"Expected dropout=0.4, got {model.dropout}"
    assert model.gaussian_noise.get('enabled') == True, "Gaussian noise not enabled"
    assert model.swa_cfg.get('enabled') == True, "SWA not enabled"
    
    model.train(X, y)
    preds = model.predict(X[:10])
    
    print(f"Predictions: {preds[:5]}")
    print("✓ Transformer config loading test passed!")
    return True

if __name__ == '__main__':
    try:
        print("Starting breast cancer model improvement tests...")
        test_mlp_silu_activation()
        test_mlp_batchnorm()
        test_mlp_gaussian_noise()
        test_mlp_swa()
        test_mlp_config_from_yaml()
        test_transformer_gaussian_noise_and_swa()
        test_transformer_config_from_yaml()
        print("\n" + "="*80)
        print("All breast cancer tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
