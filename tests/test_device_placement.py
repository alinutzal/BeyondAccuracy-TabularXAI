"""
Test that deep learning models are correctly placed on GPU/CPU device.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models.deep_learning import MLPClassifier, TransformerClassifier, LIGHTNING_AVAILABLE


def test_mlp_device_placement_standard():
    """Test MLP device placement with standard PyTorch training."""
    print("\n" + "="*80)
    print("Testing MLP device placement (standard PyTorch)")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)
    
    # Train MLP without Lightning
    mlp = MLPClassifier(
        hidden_dims=[16, 8],
        dropout=0.2,
        training={'batch_size': 32, 'epochs': 2},
        random_seed=42,
        use_lightning=False
    )
    
    mlp.train(X_train_df, y_train_series)
    
    # Check device placement
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_device = str(next(mlp.model.parameters()).device).split(':')[0]
    
    print(f"  Expected device: {expected_device}")
    print(f"  Model device: {model_device}")
    print(f"  self.device: {mlp.device}")
    
    assert model_device == expected_device, f"Model on {model_device}, expected {expected_device}"
    assert mlp.device == expected_device, f"self.device is {mlp.device}, expected {expected_device}"
    
    print("✓ MLP device placement correct (standard PyTorch)")
    return True


def test_mlp_device_placement_lightning():
    """Test MLP device placement with Lightning training."""
    if not LIGHTNING_AVAILABLE:
        print("\n" + "="*80)
        print("Skipping Lightning device test - PyTorch Lightning not available")
        print("="*80)
        return True
    
    print("\n" + "="*80)
    print("Testing MLP device placement (PyTorch Lightning)")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)
    
    # Train MLP with Lightning
    mlp = MLPClassifier(
        hidden_dims=[16, 8],
        dropout=0.2,
        training={'batch_size': 32, 'epochs': 2},
        random_seed=42,
        use_lightning=True
    )
    
    mlp.train(X_train_df, y_train_series)
    
    # Check device placement
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_device = str(next(mlp.model.parameters()).device).split(':')[0]
    
    print(f"  Expected device: {expected_device}")
    print(f"  Model device: {model_device}")
    print(f"  self.device: {mlp.device}")
    
    assert model_device == expected_device, f"Model on {model_device}, expected {expected_device}"
    assert mlp.device == expected_device, f"self.device is {mlp.device}, expected {expected_device}"
    
    print("✓ MLP device placement correct (PyTorch Lightning)")
    return True


def test_transformer_device_placement_standard():
    """Test Transformer device placement with standard PyTorch training."""
    print("\n" + "="*80)
    print("Testing Transformer device placement (standard PyTorch)")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)
    
    # Train Transformer without Lightning
    transformer = TransformerClassifier(
        d_model=16,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 2},
        random_seed=42,
        use_lightning=False
    )
    
    transformer.train(X_train_df, y_train_series)
    
    # Check device placement
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_device = str(next(transformer.model.parameters()).device).split(':')[0]
    
    print(f"  Expected device: {expected_device}")
    print(f"  Model device: {model_device}")
    print(f"  self.device: {transformer.device}")
    
    assert model_device == expected_device, f"Model on {model_device}, expected {expected_device}"
    assert transformer.device == expected_device, f"self.device is {transformer.device}, expected {expected_device}"
    
    print("✓ Transformer device placement correct (standard PyTorch)")
    return True


def test_transformer_device_placement_lightning():
    """Test Transformer device placement with Lightning training."""
    if not LIGHTNING_AVAILABLE:
        print("\n" + "="*80)
        print("Skipping Lightning device test - PyTorch Lightning not available")
        print("="*80)
        return True
    
    print("\n" + "="*80)
    print("Testing Transformer device placement (PyTorch Lightning)")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)
    
    # Train Transformer with Lightning
    transformer = TransformerClassifier(
        d_model=16,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 2},
        random_seed=42,
        use_lightning=True
    )
    
    transformer.train(X_train_df, y_train_series)
    
    # Check device placement
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_device = str(next(transformer.model.parameters()).device).split(':')[0]
    
    print(f"  Expected device: {expected_device}")
    print(f"  Model device: {model_device}")
    print(f"  self.device: {transformer.device}")
    
    assert model_device == expected_device, f"Model on {model_device}, expected {expected_device}"
    assert transformer.device == expected_device, f"self.device is {transformer.device}, expected {expected_device}"
    
    print("✓ Transformer device placement correct (PyTorch Lightning)")
    return True


if __name__ == '__main__':
    try:
        print("Testing device placement for deep learning models...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Lightning available: {LIGHTNING_AVAILABLE}")
        
        test_mlp_device_placement_standard()
        test_mlp_device_placement_lightning()
        test_transformer_device_placement_standard()
        test_transformer_device_placement_lightning()
        
        print("\n" + "="*80)
        print("All device placement tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
