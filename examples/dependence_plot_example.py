"""
Example demonstrating SHAP dependence plots.

Dependence plots show how a single feature affects the model's predictions,
with the feature value on the x-axis and the SHAP value on the y-axis.
Points are colored by another feature to reveal interactions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import XGBoostClassifier
from explainability import SHAPExplainer


def main():
    print("="*80)
    print("SHAP Dependence Plot Example")
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
    
    test_metrics = model.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1 Score: {test_metrics['f1_score']:.4f}")
    
    # 3. Generate SHAP Explanations
    print("\n3. Generating SHAP explanations...")
    shap_explainer = SHAPExplainer(model, X_train, model_type='tree')
    _ = shap_explainer.explain(X_test.head(100))
    
    # Get feature importance
    shap_importance = shap_explainer.get_feature_importance(X_test.head(100))
    print("\n   Top 5 Most Important Features:")
    for idx, row in shap_importance.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # 4. Create Dependence Plots
    print("\n4. Creating SHAP dependence plots...")
    
    # Get top 3 features
    top_features = shap_importance.head(3)['feature'].tolist()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'dependence_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n   Saving plots to: {output_dir}/")
    
    for i, feature in enumerate(top_features, 1):
        print(f"\n   {i}. Creating dependence plot for '{feature}'...")
        
        # Auto-detect interaction
        safe_name = feature.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f'{i}_dependence_{safe_name}_auto.png')
        shap_explainer.plot_dependence(
            feature,
            X_test.head(100),
            interaction_index='auto',
            save_path=output_path
        )
        print(f"      ✓ Saved: {i}_dependence_{safe_name}_auto.png")
        
        # With specific interaction (using the next most important feature)
        if i < len(top_features):
            interaction_feature = top_features[i]  # Next feature in the list
            safe_interaction = interaction_feature.replace(' ', '_').replace('/', '_')
            output_path_int = os.path.join(output_dir, f'{i}_dependence_{safe_name}_vs_{safe_interaction}.png')
            shap_explainer.plot_dependence(
                feature,
                X_test.head(100),
                interaction_index=interaction_feature,
                save_path=output_path_int
            )
            print(f"      ✓ Saved: {i}_dependence_{safe_name}_vs_{safe_interaction}.png")
    
    # 5. Understanding Dependence Plots
    print("\n" + "="*80)
    print("Understanding SHAP Dependence Plots")
    print("="*80)
    print("""
Dependence plots show:
- X-axis: Feature value
- Y-axis: SHAP value (impact on model prediction)
- Color: Another feature's value (to show interactions)

What to look for:
- Linear relationships: Suggests linear effect on predictions
- Non-linear patterns: Indicates complex feature effects
- Vertical spread: Shows uncertainty or interaction effects
- Color patterns: Reveals feature interactions

For example, if 'worst concave points' shows:
- Positive slope: Higher values increase predicted probability
- Color variation: Interactions with other features affect the relationship
""")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}/")
    print("You can now:")
    print("1. View the generated plots to understand feature effects")
    print("2. Compare automatic vs. manual interaction selections")
    print("3. Use these plots to explain model predictions to stakeholders")


if __name__ == '__main__':
    main()
