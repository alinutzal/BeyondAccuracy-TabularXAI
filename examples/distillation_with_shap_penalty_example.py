"""
Example demonstrating XGBoost → Deep Learning distillation with SHAP-based consistency penalty.

This example shows how to:
1. Train an XGBoost model (teacher)
2. Get SHAP feature importance to identify top-k features
3. Train a deep learning model with distillation + consistency penalty on top-k features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import shap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import XGBoostClassifier, MLPClassifier


def main():
    print("\n" + "="*80)
    print("XGBoost → DL Distillation with SHAP Consistency Penalty")
    print("="*80)
    
    # 1. Create a synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.05
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # 2. Train XGBoost as teacher model
    print("\n2. Training XGBoost teacher model...")
    xgb_teacher = XGBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_teacher.train(X_train, y_train)
    
    # Evaluate teacher
    xgb_metrics = xgb_teacher.evaluate(X_test, y_test)
    print(f"   XGBoost Test Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"   XGBoost Test F1 Score: {xgb_metrics['f1_score']:.4f}")
    
    # 3. Get SHAP feature importance to identify top-k features
    print("\n3. Computing SHAP feature importance...")
    explainer = shap.TreeExplainer(xgb_teacher.model)
    shap_values = explainer.shap_values(X_train)
    
    # Get mean absolute SHAP values for each feature
    if isinstance(shap_values, list):
        # For multi-class, average across classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top-k feature indices
    top_k = 5
    top_k_indices = np.argsort(mean_abs_shap)[-top_k:].tolist()
    top_k_feature_names = [X.columns[i] for i in top_k_indices]
    
    print(f"   Top {top_k} features by SHAP importance:")
    for i, (idx, name) in enumerate(zip(top_k_indices, top_k_feature_names)):
        print(f"      {i+1}. {name} (index {idx}): {mean_abs_shap[idx]:.4f}")
    
    # 4. Get soft probabilities from teacher
    print("\n4. Extracting soft probabilities from teacher...")
    teacher_train_probs = xgb_teacher.predict_proba(X_train)
    teacher_train_logits = np.log(teacher_train_probs + 1e-10)
    
    # 5. Train baseline MLP with distillation (no consistency penalty)
    print("\n5. Training MLP with distillation (no consistency penalty)...")
    mlp_baseline = MLPClassifier(
        hidden_dims=[64, 32],
        dropout=0.2,
        weight_decay=1e-4,
        optimizer={'name': 'AdamW', 'lr': 0.001},
        training={'batch_size': 64, 'epochs': 30},
        distillation={
            'enabled': True,
            'lambda': 0.7,
            'temperature': 2.0
        },
        random_seed=42
    )
    mlp_baseline.train(X_train, y_train, teacher_probs=teacher_train_logits)
    
    # Evaluate baseline MLP
    mlp_baseline_metrics = mlp_baseline.evaluate(X_test, y_test)
    print(f"   Baseline (no penalty) Test Accuracy: {mlp_baseline_metrics['accuracy']:.4f}")
    print(f"   Baseline (no penalty) Test F1 Score: {mlp_baseline_metrics['f1_score']:.4f}")
    
    # 6. Train MLP with distillation + SHAP consistency penalty
    print("\n6. Training MLP with distillation + SHAP consistency penalty...")
    print(f"   Using top-{top_k} features for consistency penalty")
    mlp_with_penalty = MLPClassifier(
        hidden_dims=[64, 32],
        dropout=0.2,
        weight_decay=1e-4,
        optimizer={'name': 'AdamW', 'lr': 0.001},
        training={'batch_size': 64, 'epochs': 30},
        distillation={
            'enabled': True,
            'lambda': 0.7,
            'temperature': 2.0,
            'consistency_penalty': {
                'enabled': True,
                'top_k_features': top_k_indices,
                'weight': 0.01  # Small weight for Jacobian penalty
            }
        },
        random_seed=42
    )
    mlp_with_penalty.train(X_train, y_train, teacher_probs=teacher_train_logits)
    
    # Evaluate MLP with penalty
    mlp_with_penalty_metrics = mlp_with_penalty.evaluate(X_test, y_test)
    print(f"   With penalty Test Accuracy: {mlp_with_penalty_metrics['accuracy']:.4f}")
    print(f"   With penalty Test F1 Score: {mlp_with_penalty_metrics['f1_score']:.4f}")
    
    # 7. Compare results
    print("\n" + "="*80)
    print("Performance Comparison Summary")
    print("="*80)
    print(f"\nTeacher (XGBoost):")
    print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {xgb_metrics['f1_score']:.4f}")
    
    print(f"\nDistilled MLP (no consistency penalty):")
    print(f"  Accuracy: {mlp_baseline_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {mlp_baseline_metrics['f1_score']:.4f}")
    
    print(f"\nDistilled MLP (with SHAP consistency penalty):")
    print(f"  Accuracy: {mlp_with_penalty_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {mlp_with_penalty_metrics['f1_score']:.4f}")
    
    acc_improvement = (mlp_with_penalty_metrics['accuracy'] - mlp_baseline_metrics['accuracy']) * 100
    f1_improvement = (mlp_with_penalty_metrics['f1_score'] - mlp_baseline_metrics['f1_score']) * 100
    print(f"\nImprovement with consistency penalty:")
    print(f"  Accuracy: {acc_improvement:+.2f}%")
    print(f"  F1 Score: {f1_improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("Key Insights:")
    print("="*80)
    print("""
1. SHAP identifies the most important features from the teacher model
2. Consistency penalty adds a Jacobian norm term for these top-k features
3. This encourages the student model to be more sensitive to important features
4. The penalty is negative in the loss (encourages larger gradients = more sensitivity)
5. Small penalty weight (0.01) is usually sufficient
6. This can provide additional improvements beyond standard distillation
""")
    
    print("✓ Example completed successfully!")


if __name__ == '__main__':
    main()
