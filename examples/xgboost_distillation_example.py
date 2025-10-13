"""
Example demonstrating XGBoost → Deep Learning distillation.

This example shows how to:
1. Train an XGBoost model to get soft probabilities (teacher)
2. Train a deep learning model (MLP or Transformer) using knowledge distillation
3. Compare performance between baseline DL and distilled DL
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import XGBoostClassifier, MLPClassifier, TransformerClassifier


def main():
    print("\n" + "="*80)
    print("XGBoost → Deep Learning Distillation Example")
    print("="*80)
    
    # 1. Create a synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.05
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(30)])
    y = pd.Series(y)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # 2. Train XGBoost as teacher model
    print("\n2. Training XGBoost teacher model...")
    xgb_teacher = XGBoostClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_teacher.train(X_train, y_train)
    
    # Evaluate teacher
    xgb_metrics = xgb_teacher.evaluate(X_test, y_test)
    print(f"   XGBoost Test Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"   XGBoost Test F1 Score: {xgb_metrics['f1_score']:.4f}")
    if 'roc_auc' in xgb_metrics and xgb_metrics['roc_auc'] is not None:
        print(f"   XGBoost Test ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    
    # Get soft probabilities from teacher
    print("\n3. Extracting soft probabilities from teacher...")
    teacher_train_probs = xgb_teacher.predict_proba(X_train)
    teacher_test_probs = xgb_teacher.predict_proba(X_test)
    print(f"   Teacher train probs shape: {teacher_train_probs.shape}")
    print(f"   Teacher test probs shape: {teacher_test_probs.shape}")
    
    # 4. Train baseline MLP (without distillation)
    print("\n4. Training baseline MLP (without distillation)...")
    mlp_baseline = MLPClassifier(
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        weight_decay=1e-4,
        optimizer={'name': 'AdamW', 'lr': 0.001},
        training={'batch_size': 64, 'epochs': 50},
        random_seed=42
    )
    mlp_baseline.train(X_train, y_train)
    
    # Evaluate baseline MLP
    mlp_baseline_metrics = mlp_baseline.evaluate(X_test, y_test)
    print(f"   Baseline MLP Test Accuracy: {mlp_baseline_metrics['accuracy']:.4f}")
    print(f"   Baseline MLP Test F1 Score: {mlp_baseline_metrics['f1_score']:.4f}")
    if 'roc_auc' in mlp_baseline_metrics and mlp_baseline_metrics['roc_auc'] is not None:
        print(f"   Baseline MLP Test ROC-AUC: {mlp_baseline_metrics['roc_auc']:.4f}")
    
    # 5. Train distilled MLP (with knowledge distillation)
    print("\n5. Training distilled MLP (with knowledge distillation)...")
    print("   Using λ=0.7, temperature=2.0")
    mlp_distilled = MLPClassifier(
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        weight_decay=1e-4,
        optimizer={'name': 'AdamW', 'lr': 0.001},
        training={'batch_size': 64, 'epochs': 50},
        distillation={
            'enabled': True,
            'lambda': 0.7,  # Weight for distillation loss
            'temperature': 2.0  # Temperature for soft targets
        },
        random_seed=42
    )
    # Pass teacher probabilities as logits (before softmax)
    # For XGBoost, we already have probabilities, so we convert them back to logits
    teacher_train_logits = np.log(teacher_train_probs + 1e-10)
    mlp_distilled.train(X_train, y_train, teacher_probs=teacher_train_logits)
    
    # Evaluate distilled MLP
    mlp_distilled_metrics = mlp_distilled.evaluate(X_test, y_test)
    print(f"   Distilled MLP Test Accuracy: {mlp_distilled_metrics['accuracy']:.4f}")
    print(f"   Distilled MLP Test F1 Score: {mlp_distilled_metrics['f1_score']:.4f}")
    if 'roc_auc' in mlp_distilled_metrics and mlp_distilled_metrics['roc_auc'] is not None:
        print(f"   Distilled MLP Test ROC-AUC: {mlp_distilled_metrics['roc_auc']:.4f}")
    
    # 6. Train baseline Transformer (without distillation)
    print("\n6. Training baseline Transformer (without distillation)...")
    transformer_baseline = TransformerClassifier(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        weight_decay=1e-4,
        optimizer={'name': 'AdamW', 'lr': 0.001},
        training={'batch_size': 64, 'epochs': 50},
        random_seed=42
    )
    transformer_baseline.train(X_train, y_train)
    
    # Evaluate baseline Transformer
    transformer_baseline_metrics = transformer_baseline.evaluate(X_test, y_test)
    print(f"   Baseline Transformer Test Accuracy: {transformer_baseline_metrics['accuracy']:.4f}")
    print(f"   Baseline Transformer Test F1 Score: {transformer_baseline_metrics['f1_score']:.4f}")
    if 'roc_auc' in transformer_baseline_metrics and transformer_baseline_metrics['roc_auc'] is not None:
        print(f"   Baseline Transformer Test ROC-AUC: {transformer_baseline_metrics['roc_auc']:.4f}")
    
    # 7. Train distilled Transformer (with knowledge distillation)
    print("\n7. Training distilled Transformer (with knowledge distillation)...")
    print("   Using λ=0.7, temperature=2.0")
    transformer_distilled = TransformerClassifier(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        weight_decay=1e-4,
        optimizer={'name': 'AdamW', 'lr': 0.001},
        training={'batch_size': 64, 'epochs': 50},
        distillation={
            'enabled': True,
            'lambda': 0.7,
            'temperature': 2.0
        },
        random_seed=42
    )
    transformer_distilled.train(X_train, y_train, teacher_probs=teacher_train_logits)
    
    # Evaluate distilled Transformer
    transformer_distilled_metrics = transformer_distilled.evaluate(X_test, y_test)
    print(f"   Distilled Transformer Test Accuracy: {transformer_distilled_metrics['accuracy']:.4f}")
    print(f"   Distilled Transformer Test F1 Score: {transformer_distilled_metrics['f1_score']:.4f}")
    if 'roc_auc' in transformer_distilled_metrics and transformer_distilled_metrics['roc_auc'] is not None:
        print(f"   Distilled Transformer Test ROC-AUC: {transformer_distilled_metrics['roc_auc']:.4f}")
    
    # 8. Compare results
    print("\n" + "="*80)
    print("Performance Comparison Summary")
    print("="*80)
    print(f"\nTeacher (XGBoost):")
    print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {xgb_metrics['f1_score']:.4f}")
    
    print(f"\nBaseline MLP:")
    print(f"  Accuracy: {mlp_baseline_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {mlp_baseline_metrics['f1_score']:.4f}")
    
    print(f"\nDistilled MLP:")
    print(f"  Accuracy: {mlp_distilled_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {mlp_distilled_metrics['f1_score']:.4f}")
    acc_improvement = (mlp_distilled_metrics['accuracy'] - mlp_baseline_metrics['accuracy']) * 100
    f1_improvement = (mlp_distilled_metrics['f1_score'] - mlp_baseline_metrics['f1_score']) * 100
    print(f"  Improvement over baseline: Acc {acc_improvement:+.2f}%, F1 {f1_improvement:+.2f}%")
    
    print(f"\nBaseline Transformer:")
    print(f"  Accuracy: {transformer_baseline_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {transformer_baseline_metrics['f1_score']:.4f}")
    
    print(f"\nDistilled Transformer:")
    print(f"  Accuracy: {transformer_distilled_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {transformer_distilled_metrics['f1_score']:.4f}")
    acc_improvement_tf = (transformer_distilled_metrics['accuracy'] - transformer_baseline_metrics['accuracy']) * 100
    f1_improvement_tf = (transformer_distilled_metrics['f1_score'] - transformer_baseline_metrics['f1_score']) * 100
    print(f"  Improvement over baseline: Acc {acc_improvement_tf:+.2f}%, F1 {f1_improvement_tf:+.2f}%")
    
    print("\n" + "="*80)
    print("Key Insights:")
    print("="*80)
    print("""
1. Knowledge Distillation transfers soft probabilities from XGBoost to DL models
2. The distilled model learns not just correct predictions, but also the
   uncertainty and confidence patterns from the teacher
3. Using λ=0.7 balances between matching teacher's soft targets (KL loss) and
   fitting ground truth labels (CE loss)
4. Temperature T=2.0 softens the probability distribution, making knowledge
   transfer more effective
5. This approach can provide a "biggest single boost" to DL model performance
""")
    
    print("✓ Example completed successfully!")


if __name__ == '__main__':
    main()
