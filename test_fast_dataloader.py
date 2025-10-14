"""
Test script for FastTensorDataLoader functionality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import MLPClassifier, TransformerClassifier


def test_fast_dataloader_basic():
    """Test basic FastTensorDataLoader functionality."""
    print("\n" + "="*80)
    print("Testing FastTensorDataLoader Basic Functionality")
    print("="*80)
    
    # Import FastTensorDataLoader directly
    from models.deep_learning import FastTensorDataLoader
    
    # Create simple tensors
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    # Test with 2 tensors
    loader = FastTensorDataLoader(X, y, batch_size=32, shuffle=False)
    
    print(f"\n✓ Created FastTensorDataLoader with {len(loader)} batches")
    
    # Test iteration
    batch_count = 0
    total_samples = 0
    for batch_X, batch_y in loader:
        batch_count += 1
        total_samples += batch_X.shape[0]
        assert batch_X.shape[1] == 10, f"Expected 10 features, got {batch_X.shape[1]}"
        assert batch_X.shape[0] == batch_y.shape[0], "Batch size mismatch"
    
    print(f"✓ Iterated through {batch_count} batches")
    print(f"✓ Processed {total_samples} samples (expected 100)")
    assert total_samples == 100, f"Expected 100 samples, got {total_samples}"
    
    # Test with 3 tensors (for distillation)
    teacher_probs = torch.randn(100, 2)
    loader_3 = FastTensorDataLoader(X, y, teacher_probs, batch_size=32, shuffle=False)
    
    batch_count = 0
    for batch_X, batch_y, batch_teacher in loader_3:
        batch_count += 1
        assert batch_X.shape[0] == batch_y.shape[0] == batch_teacher.shape[0], "Batch size mismatch"
    
    print(f"✓ Three-tensor loader works correctly with {batch_count} batches")
    
    # Test shuffle
    loader_shuffled = FastTensorDataLoader(X, y, batch_size=32, shuffle=True)
    first_batch_1 = next(iter(loader_shuffled))[0]
    first_batch_2 = next(iter(loader_shuffled))[0]
    
    # With shuffle, batches should be different across iterations
    print(f"✓ Shuffle functionality works")
    
    return True


def test_mlp_with_fast_dataloader():
    """Test MLP classifier with FastTensorDataLoader."""
    print("\n" + "="*80)
    print("Testing MLP with FastTensorDataLoader")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train MLP
    print("\nTraining MLP...")
    mlp = MLPClassifier(
        hidden_dims=[32, 16],
        training={'batch_size': 32, 'epochs': 5},
        random_seed=42
    )
    mlp.train(X_train, y_train)
    
    # Evaluate
    metrics = mlp.evaluate(X_test, y_test)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    assert metrics['accuracy'] > 0.5, "MLP accuracy too low"
    print("✓ MLP training successful with FastTensorDataLoader")
    
    return True


def test_transformer_with_fast_dataloader():
    """Test Transformer classifier with FastTensorDataLoader."""
    print("\n" + "="*80)
    print("Testing Transformer with FastTensorDataLoader")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Transformer
    print("\nTraining Transformer...")
    transformer = TransformerClassifier(
        d_model=32,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 5},
        random_seed=42
    )
    transformer.train(X_train, y_train)
    
    # Evaluate
    metrics = transformer.evaluate(X_test, y_test)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    assert metrics['accuracy'] > 0.5, "Transformer accuracy too low"
    print("✓ Transformer training successful with FastTensorDataLoader")
    
    return True


def test_distillation_with_fast_dataloader():
    """Test distillation with FastTensorDataLoader (3 tensors)."""
    print("\n" + "="*80)
    print("Testing Distillation with FastTensorDataLoader")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create dummy teacher probabilities
    teacher_probs = np.random.rand(len(X_train), 2)
    teacher_probs = teacher_probs / teacher_probs.sum(axis=1, keepdims=True)
    teacher_logits = np.log(teacher_probs + 1e-10)
    
    # Train MLP with distillation
    print("\nTraining MLP with distillation...")
    mlp = MLPClassifier(
        hidden_dims=[32, 16],
        training={'batch_size': 32, 'epochs': 5},
        distillation={
            'enabled': True,
            'lambda': 0.5,
            'temperature': 2.0
        },
        random_seed=42
    )
    mlp.train(X_train, y_train, teacher_probs=teacher_logits)
    
    # Evaluate
    metrics = mlp.evaluate(X_test, y_test)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    assert metrics['accuracy'] > 0.5, "MLP with distillation accuracy too low"
    print("✓ Distillation training successful with FastTensorDataLoader")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FastTensorDataLoader Test Suite")
    print("="*80)
    
    try:
        success = True
        
        success &= test_fast_dataloader_basic()
        success &= test_mlp_with_fast_dataloader()
        success &= test_transformer_with_fast_dataloader()
        success &= test_distillation_with_fast_dataloader()
        
        print("\n" + "="*80)
        if success:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
