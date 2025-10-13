"""
Example: Training MLP and Transformer on Breast Cancer Dataset

This example demonstrates the improvements for tiny datasets:
- SiLU activation
- BatchNorm for better performance on small datasets
- Gaussian noise augmentation (σ=0.01)
- SWA (Stochastic Weight Averaging) for final epochs
- Label smoothing
- Repeated stratified CV for stable estimates
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models.deep_learning import MLPClassifier, TransformerClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import yaml


def run_breast_cancer_mlp():
    """Run MLP on breast cancer dataset with optimized configuration."""
    print("="*80)
    print("Breast Cancer MLP Example")
    print("="*80)
    
    # Load data
    print("\n1. Loading Breast Cancer dataset...")
    loader = DataLoader('breast_cancer', random_state=42)
    X, y = loader.load_data()
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Load optimized config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'breast_cancer_mlp.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Remove non-model keys
    config.pop('notes', None)
    
    # Reduce epochs for demo
    config['training']['epochs'] = 50
    
    print("\n2. MLP Configuration:")
    print(f"   Architecture: {config['hidden_dims']}")
    print(f"   Activation: {config['activation']}")
    print(f"   BatchNorm: {config['batch_norm_per_block']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Gaussian Noise: σ={config['gaussian_noise']['std']}")
    print(f"   SWA: final {config['swa']['final_epochs']} epochs")
    print(f"   Label Smoothing: {config['label_smoothing']}")
    
    # Perform repeated stratified 5-fold CV (reduced for demo)
    print("\n3. Running 2×5 Repeated Stratified CV...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Standardize per fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert back to DataFrame
        import pandas as pd
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
        
        # Train model
        model = MLPClassifier(**config)
        model.train(X_train_scaled, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_val_scaled, y_val)
        cv_scores.append(metrics['accuracy'])
        print(f"   Fold {fold+1}: Accuracy = {metrics['accuracy']:.4f}")
    
    print(f"\n4. Results:")
    print(f"   Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"   Min: {np.min(cv_scores):.4f}, Max: {np.max(cv_scores):.4f}")


def run_breast_cancer_transformer():
    """Run Transformer on breast cancer dataset with optimized configuration."""
    print("\n" + "="*80)
    print("Breast Cancer Transformer Example")
    print("="*80)
    
    # Load data
    print("\n1. Loading Breast Cancer dataset...")
    loader = DataLoader('breast_cancer', random_state=42)
    X, y = loader.load_data()
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Load optimized config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'breast_cancer_transformer.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Remove non-model keys
    config.pop('notes', None)
    
    # Reduce epochs for demo
    config['training']['epochs'] = 50
    
    print("\n2. Transformer Configuration:")
    print(f"   d_model: {config['d_model']}")
    print(f"   Heads: {config['nhead']}")
    print(f"   Layers: {config['num_layers']}")
    print(f"   FFN: {config['dim_feedforward']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Gaussian Noise: σ={config['gaussian_noise']['std']}")
    print(f"   SWA: final {config['swa']['final_epochs']} epochs")
    print(f"   Label Smoothing: {config['label_smoothing']}")
    
    # Single train/test split for demo (full CV would take longer)
    print("\n3. Training on single 80/20 split...")
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Train model
    model = TransformerClassifier(**config)
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\n4. Results:")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test F1 Score: {metrics['f1_score']:.4f}")
    print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")


def main():
    print("\n" + "="*80)
    print("Breast Cancer Dataset - MLP and Transformer Examples")
    print("="*80)
    print("\nDataset Characteristics:")
    print("  - Tiny dataset: n=569 samples")
    print("  - All numerical features: 30 features")
    print("  - High signal, binary classification")
    print("  - Easy to overfit without proper regularization")
    print("\nOptimizations Applied:")
    print("  - Small architecture to prevent overfitting")
    print("  - Heavy regularization (dropout=0.4, weight_decay=5e-4)")
    print("  - BatchNorm for better training on small datasets")
    print("  - Gaussian noise augmentation (σ=0.01)")
    print("  - SWA (Stochastic Weight Averaging) for last 30 epochs")
    print("  - Label smoothing (0.02)")
    print("  - Cosine learning rate schedule with warmup")
    print("\n")
    
    run_breast_cancer_mlp()
    run_breast_cancer_transformer()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\nKey Takeaways:")
    print("  - MLP performs well on this dataset with proper configuration")
    print("  - Transformer kept tiny (d_model=128, 2 layers) to avoid overfitting")
    print("  - Expect parity between MLP and Transformer, not dominance")
    print("  - Repeated stratified CV provides stable performance estimates")
    print("  - Noise injection + SWA moves the needle for final performance")
    print("\n")


if __name__ == '__main__':
    main()
