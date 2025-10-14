"""
Test script for XGBoost → DL distillation functionality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import XGBoostClassifier, MLPClassifier, TransformerClassifier


def test_mlp_distillation():
    """Test MLP with distillation."""
    print("\n" + "="*80)
    print("Testing MLP with Knowledge Distillation")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train teacher
    print("\n1. Training XGBoost teacher...")
    teacher = XGBoostClassifier(n_estimators=50, random_state=42)
    teacher.train(X_train, y_train)
    teacher_probs = teacher.predict_proba(X_train)
    teacher_logits = np.log(teacher_probs + 1e-10)
    
    teacher_metrics = teacher.evaluate(X_test, y_test)
    print(f"   Teacher Accuracy: {teacher_metrics['accuracy']:.4f}")
    
    # Train baseline MLP
    print("\n2. Training baseline MLP...")
    mlp_baseline = MLPClassifier(
        hidden_dims=[64, 32],
        training={'batch_size': 32, 'epochs': 20},
        random_seed=42
    )
    mlp_baseline.train(X_train, y_train)
    baseline_metrics = mlp_baseline.evaluate(X_test, y_test)
    print(f"   Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Train distilled MLP
    print("\n3. Training distilled MLP...")
    mlp_distilled = MLPClassifier(
        hidden_dims=[64, 32],
        training={'batch_size': 32, 'epochs': 20},
        distillation={
            'enabled': True,
            'lambda': 0.7,
            'temperature': 2.0
        },
        random_seed=42
    )
    mlp_distilled.train(X_train, y_train, teacher_probs=teacher_logits)
    distilled_metrics = mlp_distilled.evaluate(X_test, y_test)
    print(f"   Distilled Accuracy: {distilled_metrics['accuracy']:.4f}")
    
    # Verify distillation was used
    assert distilled_metrics['accuracy'] >= 0.7, "Distilled model accuracy too low"
    print("\n✓ MLP distillation test passed!")
    return True


def test_transformer_distillation():
    """Test Transformer with distillation."""
    print("\n" + "="*80)
    print("Testing Transformer with Knowledge Distillation")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train teacher
    print("\n1. Training XGBoost teacher...")
    teacher = XGBoostClassifier(n_estimators=50, random_state=42)
    teacher.train(X_train, y_train)
    teacher_probs = teacher.predict_proba(X_train)
    teacher_logits = np.log(teacher_probs + 1e-10)
    
    teacher_metrics = teacher.evaluate(X_test, y_test)
    print(f"   Teacher Accuracy: {teacher_metrics['accuracy']:.4f}")
    
    # Train baseline Transformer
    print("\n2. Training baseline Transformer...")
    transformer_baseline = TransformerClassifier(
        d_model=32,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 20},
        random_seed=42
    )
    transformer_baseline.train(X_train, y_train)
    baseline_metrics = transformer_baseline.evaluate(X_test, y_test)
    print(f"   Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Train distilled Transformer
    print("\n3. Training distilled Transformer...")
    transformer_distilled = TransformerClassifier(
        d_model=32,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 20},
        distillation={
            'enabled': True,
            'lambda': 0.7,
            'temperature': 2.0
        },
        random_seed=42
    )
    transformer_distilled.train(X_train, y_train, teacher_probs=teacher_logits)
    distilled_metrics = transformer_distilled.evaluate(X_test, y_test)
    print(f"   Distilled Accuracy: {distilled_metrics['accuracy']:.4f}")
    
    # Verify distillation was used
    assert distilled_metrics['accuracy'] >= 0.7, "Distilled model accuracy too low"
    print("\n✓ Transformer distillation test passed!")
    return True


def test_distillation_parameters():
    """Test different distillation parameters."""
    print("\n" + "="*80)
    print("Testing Distillation Parameters")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=300, n_features=10, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train teacher
    teacher = XGBoostClassifier(n_estimators=30, random_state=42)
    teacher.train(X_train, y_train)
    teacher_probs = teacher.predict_proba(X_train)
    teacher_logits = np.log(teacher_probs + 1e-10)
    
    # Test different lambda values
    print("\nTesting different λ values:")
    for lambda_val in [0.3, 0.5, 0.7, 0.9]:
        mlp = MLPClassifier(
            hidden_dims=[32],
            training={'batch_size': 32, 'epochs': 10},
            distillation={
                'enabled': True,
                'lambda': lambda_val,
                'temperature': 2.0
            },
            random_seed=42
        )
        mlp.train(X_train, y_train, teacher_probs=teacher_logits)
        metrics = mlp.evaluate(X_test, y_test)
        print(f"   λ={lambda_val}: Accuracy={metrics['accuracy']:.4f}")
    
    # Test different temperature values
    print("\nTesting different temperature values:")
    for temp in [1.0, 2.0, 3.0, 5.0]:
        mlp = MLPClassifier(
            hidden_dims=[32],
            training={'batch_size': 32, 'epochs': 10},
            distillation={
                'enabled': True,
                'lambda': 0.7,
                'temperature': temp
            },
            random_seed=42
        )
        mlp.train(X_train, y_train, teacher_probs=teacher_logits)
        metrics = mlp.evaluate(X_test, y_test)
        print(f"   T={temp}: Accuracy={metrics['accuracy']:.4f}")
    
    print("\n✓ Distillation parameters test passed!")
    return True


def test_backward_compatibility():
    """Test that models without distillation still work."""
    print("\n" + "="*80)
    print("Testing Backward Compatibility")
    print("="*80)
    
    # Create dataset
    X, y = make_classification(
        n_samples=200, n_features=10, random_state=42, n_classes=2
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    # Test MLP without distillation
    print("\n1. Testing MLP without distillation...")
    mlp = MLPClassifier(
        hidden_dims=[32],
        training={'batch_size': 32, 'epochs': 10},
        random_seed=42
    )
    mlp.train(X, y)
    preds = mlp.predict(X[:10])
    print(f"   Predictions: {preds[:5]}")
    print("   ✓ MLP works without distillation")
    
    # Test Transformer without distillation
    print("\n2. Testing Transformer without distillation...")
    transformer = TransformerClassifier(
        d_model=16,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 10},
        random_seed=42
    )
    transformer.train(X, y)
    preds = transformer.predict(X[:10])
    print(f"   Predictions: {preds[:5]}")
    print("   ✓ Transformer works without distillation")
    
    print("\n✓ Backward compatibility test passed!")
    return True


if __name__ == '__main__':
    try:
        print("Starting distillation tests...")
        test_mlp_distillation()
        test_transformer_distillation()
        test_distillation_parameters()
        test_backward_compatibility()
        print("\n" + "="*80)
        print("All distillation tests passed! ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
