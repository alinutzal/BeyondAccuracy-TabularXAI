"""
Example demonstrating PyTorch Lightning integration with deep learning models.

This example shows:
1. Training MLP without Lightning (default/backward compatible)
2. Training MLP with Lightning
3. Training Transformer without Lightning
4. Training Transformer with Lightning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from models.deep_learning import MLPClassifier, TransformerClassifier, LIGHTNING_AVAILABLE


def main():
    print("=" * 80)
    print("PyTorch Lightning Integration Example")
    print("=" * 80)
    print(f"\nPyTorch Lightning available: {LIGHTNING_AVAILABLE}")
    
    # Create a synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Example 1: MLP without Lightning (default)
    print("\n2. Training MLP without Lightning (default behavior)...")
    mlp_standard = MLPClassifier(
        hidden_dims=[64, 32],
        dropout=0.2,
        training={'batch_size': 32, 'epochs': 10},
        random_seed=42,
        use_lightning=False  # Default - standard PyTorch training
    )
    mlp_standard.train(X_train_df, y_train_series)
    
    # Evaluate
    metrics_standard = mlp_standard.evaluate(X_test_df, y_test_series)
    print(f"   Accuracy: {metrics_standard['accuracy']:.4f}")
    print(f"   F1 Score: {metrics_standard['f1_score']:.4f}")
    
    # Example 2: MLP with Lightning
    if LIGHTNING_AVAILABLE:
        print("\n3. Training MLP with Lightning...")
        mlp_lightning = MLPClassifier(
            hidden_dims=[64, 32],
            dropout=0.2,
            training={'batch_size': 32, 'epochs': 10},
            random_seed=42,
            use_lightning=True  # Use PyTorch Lightning
        )
        mlp_lightning.train(X_train_df, y_train_series)
        
        # Evaluate
        metrics_lightning = mlp_lightning.evaluate(X_test_df, y_test_series)
        print(f"   Accuracy: {metrics_lightning['accuracy']:.4f}")
        print(f"   F1 Score: {metrics_lightning['f1_score']:.4f}")
    else:
        print("\n3. Skipping Lightning example - PyTorch Lightning not available")
    
    # Example 3: Transformer without Lightning
    print("\n4. Training Transformer without Lightning...")
    transformer_standard = TransformerClassifier(
        d_model=32,
        nhead=2,
        num_layers=1,
        training={'batch_size': 32, 'epochs': 10},
        random_seed=42,
        use_lightning=False
    )
    transformer_standard.train(X_train_df, y_train_series)
    
    # Evaluate
    metrics_transformer = transformer_standard.evaluate(X_test_df, y_test_series)
    print(f"   Accuracy: {metrics_transformer['accuracy']:.4f}")
    print(f"   F1 Score: {metrics_transformer['f1_score']:.4f}")
    
    # Example 4: Transformer with Lightning
    if LIGHTNING_AVAILABLE:
        print("\n5. Training Transformer with Lightning...")
        transformer_lightning = TransformerClassifier(
            d_model=32,
            nhead=2,
            num_layers=1,
            training={'batch_size': 32, 'epochs': 10},
            random_seed=42,
            use_lightning=True
        )
        transformer_lightning.train(X_train_df, y_train_series)
        
        # Evaluate
        metrics_transformer_lightning = transformer_lightning.evaluate(X_test_df, y_test_series)
        print(f"   Accuracy: {metrics_transformer_lightning['accuracy']:.4f}")
        print(f"   F1 Score: {metrics_transformer_lightning['f1_score']:.4f}")
    else:
        print("\n5. Skipping Lightning example - PyTorch Lightning not available")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nBoth standard PyTorch and PyTorch Lightning training are supported:")
    print("  • use_lightning=False (default): Standard PyTorch training loop")
    print("  • use_lightning=True: PyTorch Lightning training")
    print("\nBenefits of using Lightning:")
    print("  • Cleaner code organization")
    print("  • Better progress tracking")
    print("  • Easy to add callbacks (early stopping, checkpointing, etc.)")
    print("  • Future-ready for distributed training and mixed precision")
    print("\nBackward compatibility is maintained - all existing code works without changes!")
    print("=" * 80)


if __name__ == '__main__':
    main()
