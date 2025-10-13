"""
Example demonstrating ShapIQ (Shapley Interaction Quantification) usage.

ShapIQ extends SHAP by computing interaction effects between features,
revealing how features work together to influence predictions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import XGBoostClassifier
from explainability import ShapIQExplainer


def main():
    print("="*80)
    print("ShapIQ (Shapley Interaction Quantification) Example")
    print("="*80)
    print("\nShapIQ reveals how features interact to influence predictions,")
    print("going beyond individual feature importance to show feature synergies.\n")
    
    # 1. Load Data
    print("1. Loading Breast Cancer dataset...")
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
    
    # 3. Compute Shapley Interactions
    print("\n3. Computing Shapley interaction values...")
    print("   Note: This analyzes how features work together (interactions)")
    
    shapiq_explainer = ShapIQExplainer(
        model, 
        X_train, 
        max_order=2  # Compute pairwise interactions
    )
    
    # Explain test set (use smaller sample for performance)
    test_sample = X_test.head(30)
    
    try:
        shapiq_result = shapiq_explainer.explain(test_sample, index='k-SII')
        print(f"   ✓ Computed {len(shapiq_result['interaction_values'])} interactions")
        print(f"   ✓ Analyzed {shapiq_result['n_samples']} samples")
    except ImportError:
        print("\n   ⚠ shapiq library not installed. Falling back to approximate interactions.")
        print("   Install with: pip install shapiq")
        print("   Continuing with approximate interaction computation...\n")
        shapiq_result = shapiq_explainer.explain(test_sample, index='k-SII')
    
    # 4. Analyze Top Interactions
    print("\n4. Top Feature Interactions:")
    top_interactions = shapiq_explainer.get_interaction_strength(top_k=10)
    
    print("\n   Strongest Feature Interactions:")
    print("   " + "-"*70)
    for idx, row in top_interactions.head(5).iterrows():
        features = row['features']
        strength = row['interaction_strength']
        if isinstance(features, tuple) and len(features) == 2:
            print(f"   {features[0]:25s} ↔ {features[1]:25s} : {strength:.4f}")
        else:
            print(f"   {str(features):52s} : {strength:.4f}")
    
    # 5. Create Visualizations
    print("\n5. Creating visualizations...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'shapiq_plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Saving plots to: {output_dir}/")
    
    # Interaction network
    try:
        network_path = os.path.join(output_dir, 'interaction_network.png')
        shapiq_explainer.plot_interaction_network(top_k=10, save_path=network_path)
        print("   ✓ Saved: interaction_network.png")
    except ImportError as e:
        print(f"   ⚠ Could not create network plot: {e}")
    except Exception as e:
        print(f"   ⚠ Network plot skipped: {e}")
    
    # Interaction heatmap
    try:
        heatmap_path = os.path.join(output_dir, 'interaction_heatmap.png')
        shapiq_explainer.plot_interaction_heatmap(top_k=15, save_path=heatmap_path)
        print("   ✓ Saved: interaction_heatmap.png")
    except Exception as e:
        print(f"   ⚠ Heatmap plot skipped: {e}")
    
    # 6. Explain Single Instance
    print("\n6. Analyzing interactions for a single prediction...")
    instance = X_test.iloc[0]
    instance_explanation = shapiq_explainer.explain_instance(instance)
    
    print(f"\n   Instance features: {instance.name}")
    print("   Top 3 interactions affecting this prediction:")
    top_instance_interactions = instance_explanation['top_interactions']
    for idx, row in top_instance_interactions.head(3).iterrows():
        features = row['features']
        strength = row['interaction_strength']
        if isinstance(features, tuple) and len(features) == 2:
            print(f"   - {features[0]} + {features[1]}: {strength:.4f}")
        else:
            print(f"   - {features}: {strength:.4f}")
    
    # 7. Understanding ShapIQ
    print("\n" + "="*80)
    print("Understanding ShapIQ Results")
    print("="*80)
    print("""
ShapIQ extends SHAP by quantifying feature interactions:

1. Individual Effects (SHAP):
   - How much each feature contributes alone
   
2. Interaction Effects (ShapIQ):
   - How features work together synergistically
   - Positive interaction: features amplify each other
   - Negative interaction: features counteract each other

Use cases:
- Discover feature synergies for feature engineering
- Understand complex model behaviors
- Identify redundant or complementary features
- Debug unexpected predictions

Visualization types:
- Network plot: Shows strongest pairwise interactions
- Heatmap: Overview of all pairwise interactions
- Interaction strengths: Quantitative ranking of effects

Note: ShapIQ computations can be expensive for large datasets.
      Use sampling for initial exploration.
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
