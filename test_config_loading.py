"""
Test that models can be instantiated from YAML configuration files.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
from models.deep_learning import MLPClassifier, TransformerClassifier

def test_mlp_config():
    """Test that MLP can be instantiated from adult_income_mlp.yaml."""
    print("\n" + "="*80)
    print("Testing MLP configuration loading")
    print("="*80)
    
    config_path = 'configs/adult_income_mlp.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Remove non-model keys
    eval_config = config.pop('eval', None)
    notes = config.pop('notes', None)
    
    print(f"\nLoaded config from {config_path}")
    print(f"Config keys: {list(config.keys())}")
    
    # Create model
    model = MLPClassifier(**config)
    
    print(f"\nModel created successfully!")
    print(f"  Hidden dims: {model.hidden_dims}")
    print(f"  Activation: {model.activation}")
    print(f"  Layer norm per block: {model.layer_norm_per_block}")
    print(f"  Dropout: {model.dropout}")
    print(f"  Weight decay: {model.weight_decay}")
    print(f"  MixUp enabled: {model.mixup.get('enabled', False)}")
    print(f"  Label smoothing: {model.label_smoothing}")
    print(f"  Optimizer: {model.optimizer_cfg.get('name', 'Adam')}")
    print(f"  Scheduler: {model.scheduler_cfg.get('name', 'None')}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Epochs: {model.epochs}")
    
    print("\n✓ MLP configuration loaded successfully!")
    return True

def test_transformer_simplified_config():
    """Test that Transformer can be instantiated with simplified config."""
    print("\n" + "="*80)
    print("Testing Transformer configuration loading (simplified)")
    print("="*80)
    
    # Note: The full adult_income_transformer.yaml has features we don't support yet
    # (feature_tokenizer, etc.), so we use a simplified version
    config = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.3,
        'weight_decay': 3e-4,
        'optimizer': {
            'name': 'AdamW',
            'lr': 5e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8
        },
        'scheduler': {
            'name': 'cosine',
            'warmup_proportion': 0.08,
            'min_lr': 0.0
        },
        'training': {
            'batch_size': 64,
            'epochs': 300,
            'auto_device': True
        },
        'random_seed': 42
    }
    
    print(f"\nUsing simplified config with keys: {list(config.keys())}")
    
    # Create model
    model = TransformerClassifier(**config)
    
    print(f"\nModel created successfully!")
    print(f"  d_model: {model.d_model}")
    print(f"  nhead: {model.nhead}")
    print(f"  num_layers: {model.num_layers}")
    print(f"  dim_feedforward: {model.dim_feedforward}")
    print(f"  Dropout: {model.dropout}")
    print(f"  Weight decay: {model.weight_decay}")
    print(f"  Optimizer: {model.optimizer_cfg.get('name', 'Adam')}")
    print(f"  Scheduler: {model.scheduler_cfg.get('name', 'None')}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Epochs: {model.epochs}")
    print(f"  Random seed: {model.random_seed}")
    
    print("\n✓ Transformer configuration loaded successfully!")
    print("\nNote: The full adult_income_transformer.yaml includes advanced features")
    print("      (feature_tokenizer, stochastic_depth, SWA) that are planned for")
    print("      future implementation. Current implementation supports the core")
    print("      features needed for improved performance.")
    return True

def test_backward_compatibility():
    """Test that legacy parameter style still works."""
    print("\n" + "="*80)
    print("Testing backward compatibility (legacy parameters)")
    print("="*80)
    
    # Old-style MLP
    mlp = MLPClassifier(
        hidden_dims=[128, 64],
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    print("✓ MLP with legacy parameters works")
    
    # Old-style Transformer
    transformer = TransformerClassifier(
        d_model=64,
        nhead=4,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    print("✓ Transformer with legacy parameters works")
    
    print("\n✓ Backward compatibility maintained!")
    return True

if __name__ == '__main__':
    try:
        print("Testing configuration loading...")
        test_mlp_config()
        test_transformer_simplified_config()
        test_backward_compatibility()
        print("\n" + "="*80)
        print("All configuration tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
