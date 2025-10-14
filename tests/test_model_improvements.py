"""
Test script to verify MLP and Transformer improvements.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import yaml
from models.deep_learning import MLPClassifier, TransformerClassifier
from sklearn.datasets import make_classification

def test_mlp_config_loading():
    """Test that MLP can load and use advanced config."""
    print("\n" + "="*80)
    print("Testing MLP with advanced configuration")
    print("="*80)
    
    # Create a simple dataset
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, 
                                n_redundant=5, random_state=42, n_classes=2)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # Load config from adult_income_mlp.yaml
    config_path = 'configs/adult_income_mlp.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
        print(f"Config keys: {list(config.keys())}")
    else:
        print(f"Config file {config_path} not found, using default config")
        config = {
            'hidden_dims': [128, 64],
            'activation': 'geglu',
            'layer_norm_per_block': True,
            'dropout': 0.2,
            'weight_decay': 1e-4,
            'mixup': {'enabled': True, 'alpha': 0.2},
            'label_smoothing': 0.03,
            'optimizer': {'name': 'AdamW', 'lr': 0.001},
            'scheduler': {'name': 'cosine', 'warmup_proportion': 0.1, 'min_lr': 0.0},
            'training': {'batch_size': 32, 'epochs': 10}
        }
    
    # Remove eval config if present
    config.pop('eval', None)
    config.pop('notes', None)
    
    # Reduce epochs for testing
    if 'training' in config:
        config['training']['epochs'] = 10
    
    # Create model with config
    model = MLPClassifier(**config)
    print(f"\nMLP initialized with:")
    print(f"  Activation: {model.activation}")
    print(f"  Layer norm per block: {model.layer_norm_per_block}")
    print(f"  MixUp enabled: {model.mixup.get('enabled', False)}")
    print(f"  Label smoothing: {model.label_smoothing}")
    print(f"  Optimizer: {model.optimizer_cfg.get('name', 'Adam')}")
    print(f"  Scheduler: {model.scheduler_cfg.get('name', 'None')}")
    
    # Train the model
    print("\nTraining MLP...")
    model.train(X, y)
    
    # Make predictions
    preds = model.predict(X[:10])
    probs = model.predict_proba(X[:10])
    
    print(f"\nPredictions: {preds[:5]}")
    print(f"Probabilities shape: {probs.shape}")
    print("✓ MLP test passed!")
    return True

def test_transformer_config_loading():
    """Test that Transformer can load and use advanced config."""
    print("\n" + "="*80)
    print("Testing Transformer with advanced configuration")
    print("="*80)
    
    # Create a simple dataset
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, 
                                n_redundant=5, random_state=42, n_classes=2)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # Test with simplified config (not the full feature tokenizer from yaml)
    config = {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.2,
        'weight_decay': 1e-4,
        'mixup': {'enabled': True, 'alpha': 0.2},
        'label_smoothing': 0.03,
        'optimizer': {'name': 'AdamW', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-8},
        'scheduler': {'name': 'cosine', 'warmup_proportion': 0.1, 'min_lr': 0.0},
        'training': {'batch_size': 32, 'epochs': 10, 'auto_device': True},
        'random_seed': 42
    }
    
    print(f"Testing with config keys: {list(config.keys())}")
    
    # Create model with config
    model = TransformerClassifier(**config)
    print(f"\nTransformer initialized with:")
    print(f"  d_model: {model.d_model}")
    print(f"  nhead: {model.nhead}")
    print(f"  num_layers: {model.num_layers}")
    print(f"  Weight decay: {model.weight_decay}")
    print(f"  MixUp enabled: {model.mixup.get('enabled', False)}")
    print(f"  Label smoothing: {model.label_smoothing}")
    print(f"  Optimizer: {model.optimizer_cfg.get('name', 'Adam')}")
    print(f"  Scheduler: {model.scheduler_cfg.get('name', 'None')}")
    print(f"  Random seed: {model.random_seed}")
    
    # Train the model
    print("\nTraining Transformer...")
    model.train(X, y)
    
    # Make predictions
    preds = model.predict(X[:10])
    probs = model.predict_proba(X[:10])
    
    print(f"\nPredictions: {preds[:5]}")
    print(f"Probabilities shape: {probs.shape}")
    print("✓ Transformer test passed!")
    return True

def test_backward_compatibility():
    """Test that old code still works (backward compatibility)."""
    print("\n" + "="*80)
    print("Testing backward compatibility")
    print("="*80)
    
    # Create a simple dataset
    X, y = make_classification(n_samples=200, n_features=10, random_state=42, n_classes=2)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    # Test MLP with old-style params
    print("\nTesting MLP with legacy parameters...")
    mlp = MLPClassifier(hidden_dims=[64, 32], dropout=0.2, 
                       learning_rate=0.001, batch_size=32, epochs=5)
    mlp.train(X, y)
    preds_mlp = mlp.predict(X[:10])
    print(f"MLP predictions: {preds_mlp[:5]}")
    print("✓ MLP backward compatibility OK")
    
    # Test Transformer with old-style params
    print("\nTesting Transformer with legacy parameters...")
    transformer = TransformerClassifier(d_model=32, nhead=2, num_layers=1,
                                       learning_rate=0.001, batch_size=32, epochs=5)
    transformer.train(X, y)
    preds_tf = transformer.predict(X[:10])
    print(f"Transformer predictions: {preds_tf[:5]}")
    print("✓ Transformer backward compatibility OK")
    
    return True

if __name__ == '__main__':
    try:
        print("Starting model improvement tests...")
        test_mlp_config_loading()
        test_transformer_config_loading()
        test_backward_compatibility()
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
