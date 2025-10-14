"""
Test PyTorch Lightning integration with deep learning models.
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


def test_mlp_without_lightning():
    """Test MLP without Lightning (default behavior)."""
    print("\n" + "="*80)
    print("Testing MLP without Lightning (backward compatibility)")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    
    # Train MLP without Lightning
    mlp = MLPClassifier(
        hidden_dims=[32, 16],
        dropout=0.2,
        training={'batch_size': 32, 'epochs': 3},
        random_seed=42,
        use_lightning=False  # Explicitly disable Lightning
    )
    
    mlp.train(X_train_df, y_train_series)
    
    # Test predictions
    y_pred = mlp.predict(X_test_df)
    accuracy = (y_pred == y_test).mean()
    
    print(f"✓ MLP without Lightning trained successfully")
    print(f"  Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.5, f"Accuracy too low: {accuracy}"
    print("✓ Test passed!")
    return True


def test_mlp_with_lightning():
    """Test MLP with Lightning."""
    if not LIGHTNING_AVAILABLE:
        print("\n" + "="*80)
        print("Skipping Lightning test - PyTorch Lightning not available")
        print("="*80)
        return True
    
    print("\n" + "="*80)
    print("Testing MLP with Lightning")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    
    # Train MLP with Lightning
    mlp = MLPClassifier(
        hidden_dims=[32, 16],
        dropout=0.2,
        training={'batch_size': 32, 'epochs': 3},
        random_seed=42,
        use_lightning=True  # Enable Lightning
    )
    
    mlp.train(X_train_df, y_train_series)
    
    # Test predictions
    y_pred = mlp.predict(X_test_df)
    accuracy = (y_pred == y_test).mean()
    
    print(f"✓ MLP with Lightning trained successfully")
    print(f"  Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.5, f"Accuracy too low: {accuracy}"
    print("✓ Test passed!")
    return True


def test_transformer_without_lightning():
    """Test Transformer without Lightning (default behavior)."""
    print("\n" + "="*80)
    print("Testing Transformer without Lightning (backward compatibility)")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    
    # Train Transformer without Lightning
    transformer = TransformerClassifier(
        d_model=32,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 3},
        random_seed=42,
        use_lightning=False  # Explicitly disable Lightning
    )
    
    transformer.train(X_train_df, y_train_series)
    
    # Test predictions
    y_pred = transformer.predict(X_test_df)
    accuracy = (y_pred == y_test).mean()
    
    print(f"✓ Transformer without Lightning trained successfully")
    print(f"  Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.5, f"Accuracy too low: {accuracy}"
    print("✓ Test passed!")
    return True


def test_transformer_with_lightning():
    """Test Transformer with Lightning."""
    if not LIGHTNING_AVAILABLE:
        print("\n" + "="*80)
        print("Skipping Lightning test - PyTorch Lightning not available")
        print("="*80)
        return True
    
    print("\n" + "="*80)
    print("Testing Transformer with Lightning")
    print("="*80)
    
    # Create simple dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    
    # Train Transformer with Lightning
    transformer = TransformerClassifier(
        d_model=32,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 3},
        random_seed=42,
        use_lightning=True  # Enable Lightning
    )
    
    transformer.train(X_train_df, y_train_series)
    
    # Test predictions
    y_pred = transformer.predict(X_test_df)
    accuracy = (y_pred == y_test).mean()
    
    print(f"✓ Transformer with Lightning trained successfully")
    print(f"  Accuracy: {accuracy:.4f}")
    
    assert accuracy > 0.5, f"Accuracy too low: {accuracy}"
    print("✓ Test passed!")
    return True


if __name__ == '__main__':
    try:
        print("Testing PyTorch Lightning integration...")
        print(f"Lightning available: {LIGHTNING_AVAILABLE}")
        
        test_mlp_without_lightning()
        test_mlp_with_lightning()
        test_transformer_without_lightning()
        test_transformer_with_lightning()
        
        print("\n" + "="*80)
        print("All Lightning integration tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
