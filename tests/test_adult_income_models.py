"""
Test script to verify MLP and Transformer work with adult_income dataset.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
from utils.data_loader import DataLoader
from models.deep_learning import MLPClassifier, TransformerClassifier

def test_mlp_adult_income():
    """Test MLP on adult_income dataset with full config."""
    print("\n" + "="*80)
    print("Testing MLP on adult_income dataset")
    print("="*80)
    
    # Load data
    print("Loading adult_income dataset...")
    loader = DataLoader('adult_income', random_state=42)
    X, y = loader.load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Categorical features: {len(loader.categorical_features)}")
    print(f"Numerical features: {len(loader.numerical_features)}")
    
    # Prepare data with QuantileTransformer
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Load config
    config_path = 'configs/adult_income_mlp.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"\nLoaded config from {config_path}")
        
        # Remove non-model keys
        config.pop('eval', None)
        config.pop('notes', None)
        
        # Reduce epochs for testing
        if 'training' in config:
            config['training']['epochs'] = 20
        
        print(f"Config: {list(config.keys())}")
    else:
        print(f"Config not found: {config_path}")
        return False
    
    # Create and train model
    print(f"\nInitializing MLP with config...")
    model = MLPClassifier(**config)
    
    print(f"\nTraining MLP (20 epochs)...")
    model.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")
    
    print("\n✓ MLP adult_income test passed!")
    return True

def test_transformer_adult_income():
    """Test Transformer on adult_income dataset with simplified config."""
    print("\n" + "="*80)
    print("Testing Transformer on adult_income dataset")
    print("="*80)
    
    # Load data
    print("Loading adult_income dataset...")
    loader = DataLoader('adult_income', random_state=42)
    X, y = loader.load_data()
    print(f"Dataset shape: {X.shape}")
    
    # Prepare data with QuantileTransformer
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Use simplified config (the full transformer config has features we don't support yet)
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
            'epochs': 20,
            'auto_device': True
        },
        'random_seed': 42
    }
    
    print(f"\nUsing simplified config with keys: {list(config.keys())}")
    
    # Create and train model
    print(f"\nInitializing Transformer with config...")
    model = TransformerClassifier(**config)
    
    print(f"\nTraining Transformer (20 epochs)...")
    model.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")
    
    print("\n✓ Transformer adult_income test passed!")
    return True

if __name__ == '__main__':
    try:
        print("Starting adult_income model tests...")
        test_mlp_adult_income()
        test_transformer_adult_income()
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
