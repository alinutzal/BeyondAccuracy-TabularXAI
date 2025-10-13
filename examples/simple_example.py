"""
Simple example demonstrating the complete workflow:
1. Load data
2. Train a model
3. Generate explanations
4. Calculate interpretability metrics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import XGBoostClassifier
from explainability import SHAPExplainer
from metrics import InterpretabilityMetrics


def main():
    print("="*80)
    print("Simple BeyondAccuracy-TabularXAI Example")
    print("="*80)
    
    # 1. Load Data
    print("\n1. Loading Breast Cancer dataset...")
    loader = DataLoader('breast_cancer', random_state=42)
    X, y = loader.load_data()
    data = loader.prepare_data(X, y, test_size=0.2)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # 2. Train Model
    print("\n2. Training XGBoost model...")
    model = XGBoostClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.train(X_train, y_train)
    
    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    # 3. Generate Explanations
    print("\n3. Generating SHAP explanations...")
    shap_explainer = SHAPExplainer(model, X_train, model_type='tree')
    _ = shap_explainer.explain(X_test.head(50))
    
    # Get feature importance
    shap_importance = shap_explainer.get_feature_importance(X_test.head(50))
    print("\n   Top 5 Most Important Features (SHAP):")
    for idx, row in shap_importance.head(5).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    # Get single instance explanation
    print("\n   Explanation for first test instance:")
    instance_exp = shap_explainer.explain_instance(X_test.iloc[0])
    top_features = sorted(instance_exp.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feature, value in top_features:
        print(f"   - {feature}: {value:.4f}")
    
    # 4. Calculate Interpretability Metrics
    print("\n4. Calculating interpretability metrics...")
    
    # Run SHAP on different subsets for stability
    import numpy as np
    importance_runs = [shap_importance]
    for seed in [42, 43]:
        subset_indices = np.random.RandomState(seed).choice(
            len(X_test), 
            size=min(30, len(X_test)), 
            replace=False
        )
        X_subset = X_test.iloc[subset_indices]
        shap_explainer_subset = SHAPExplainer(model, X_train, model_type='tree')
        _ = shap_explainer_subset.explain(X_subset)
        importance_subset = shap_explainer_subset.get_feature_importance(X_subset)
        importance_runs.append(importance_subset)
    
    # Calculate stability
    stability = InterpretabilityMetrics.feature_importance_stability(
        importance_runs, 
        method='spearman'
    )
    print(f"   Feature Importance Stability: {stability:.4f}")
    
    # Get explanations for fidelity
    shap_explanations = []
    for i in range(min(30, len(X_test))):
        exp = shap_explainer.explain_instance(X_test.iloc[i])
        shap_explanations.append(exp)
    
    fidelity = InterpretabilityMetrics.explanation_fidelity(
        model, 
        X_test.head(30), 
        shap_explanations, 
        top_k=5
    )
    print(f"   Explanation Fidelity: {fidelity:.4f}")
    
    complexity = InterpretabilityMetrics.explanation_complexity(shap_explanations)
    print(f"   Average Explanation Complexity: {complexity:.2f} features")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Model Performance:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"\nInterpretability:")
    print(f"  - Feature Importance Stability: {stability:.4f}")
    print(f"  - Explanation Fidelity: {fidelity:.4f}")
    print(f"  - Explanation Complexity: {complexity:.2f} features")
    print(f"\nTop 3 Important Features:")
    for idx, row in shap_importance.head(3).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
