"""
Test GradientBoostingClassifier wrapper.
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Import directly from the file to avoid torch dependency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
from gradient_boosting import GradientBoostingClassifier


def test_gradient_boosting_classifier():
    """Test GradientBoostingClassifier wrapper."""
    print("\n" + "="*80)
    print("Testing GradientBoostingClassifier Wrapper")
    print("="*80)
    
    # Create a simple dataset
    print("\nCreating test dataset...")
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split into train and test
    split_idx = 150
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"  Train set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Test initialization
    print("\nTesting model initialization...")
    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    print(f"  Model name: {model.model_name}")
    print(f"  ✓ Model initialized successfully!")
    
    # Test training
    print("\nTesting model training...")
    model.train(X_train, y_train)
    print("  ✓ Model trained successfully!")
    
    # Test prediction
    print("\nTesting predictions...")
    y_pred = model.predict(X_test)
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Unique predictions: {np.unique(y_pred)}")
    assert len(y_pred) == len(X_test), "Prediction length mismatch"
    print("  ✓ Predictions generated successfully!")
    
    # Test probability prediction
    print("\nTesting probability predictions...")
    y_proba = model.predict_proba(X_test)
    print(f"  Probabilities shape: {y_proba.shape}")
    assert y_proba.shape == (len(X_test), 2), "Probability shape mismatch"
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    print("  ✓ Probability predictions valid!")
    
    # Test evaluation
    print("\nTesting model evaluation...")
    metrics = model.evaluate(X_test, y_test)
    print("  Metrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"    {key}: {value:.4f}")
    
    assert 'accuracy' in metrics, "Accuracy metric missing"
    assert 'f1_score' in metrics, "F1 score metric missing"
    assert metrics['accuracy'] > 0.5, "Accuracy too low"
    print("  ✓ Evaluation completed successfully!")
    
    # Test feature importance
    print("\nTesting feature importance...")
    feature_names = X.columns.tolist()
    importance = model.get_feature_importance(feature_names)
    print(f"  Feature importance shape: {importance.shape}")
    print("  Top 3 features:")
    for idx, row in importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    assert len(importance) == len(feature_names), "Feature importance length mismatch"
    assert 'feature' in importance.columns, "'feature' column missing"
    assert 'importance' in importance.columns, "'importance' column missing"
    print("  ✓ Feature importance calculated successfully!")
    
    print("\n✓ All GradientBoostingClassifier tests passed!")
    return True


def test_gradient_boosting_with_config():
    """Test GradientBoostingClassifier with config-like parameters."""
    print("\n" + "="*80)
    print("Testing GradientBoostingClassifier with Config Parameters")
    print("="*80)
    
    # Create a simple dataset
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Test with different config styles
    configs = {
        'default': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'shallow': {
            'n_estimators': 200,
            'max_depth': 2,
            'learning_rate': 0.05,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'subsample': 0.8,
            'random_state': 42
        },
        'deep': {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.03,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.9,
            'random_state': 42
        }
    }
    
    for config_name, params in configs.items():
        print(f"\n  Testing {config_name} configuration...")
        model = GradientBoostingClassifier(**params)
        model.train(X, y)
        accuracy = model.evaluate(X, y)['accuracy']
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"  ✓ {config_name} configuration works!")
    
    print("\n✓ All configuration tests passed!")
    return True


if __name__ == '__main__':
    try:
        print("\nRunning GradientBoostingClassifier tests...")
        test_gradient_boosting_classifier()
        test_gradient_boosting_with_config()
        
        print("\n" + "="*80)
        print("All GradientBoostingClassifier tests passed! ✓")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
