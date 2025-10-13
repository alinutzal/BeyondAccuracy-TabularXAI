"""
Example demonstrating SHAP waterfall plots (Shapley flow).

Waterfall plots show how each feature pushes the prediction from the base value
(expected value) to the final model prediction. This visualizes the "flow" of
Shapley values, making it easy to explain individual predictions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import XGBoostClassifier
from explainability import SHAPExplainer


def main():
    print("="*80)
    print("SHAP Waterfall Plot Example (Shapley Flow)")
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
    _ = shap_explainer.explain(X_test.head(10))
    
    # 4. Create Waterfall Plots for Different Instances
    print("\n4. Creating SHAP waterfall plots (Shapley flow)...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'waterfall_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n   Saving plots to: {output_dir}/")
    
    # Select a few interesting instances to visualize
    num_instances = min(5, len(X_test))
    
    for i in range(num_instances):
        instance = X_test.iloc[i]
        prediction = model.predict(instance.to_frame().T)[0]
        prediction_proba = model.predict_proba(instance.to_frame().T)[0]
        
        print(f"\n   {i+1}. Instance {i}:")
        print(f"      - Prediction: {prediction}")
        print(f"      - Probability: {prediction_proba:.4f}")
        
        # Create waterfall plot
        output_path = os.path.join(output_dir, f'waterfall_instance_{i}.png')
        shap_explainer.plot_waterfall(
            instance,
            class_idx=0,
            max_display=10,
            save_path=output_path
        )
        print(f"      ✓ Saved: waterfall_instance_{i}.png")
    
    # 5. Understanding Waterfall Plots
    print("\n" + "="*80)
    print("Understanding SHAP Waterfall Plots (Shapley Flow)")
    print("="*80)
    print("""
Waterfall plots show the "flow" of Shapley values:
- Base value (E[f(X)]): Expected model output (shown at bottom)
- Red bars: Features that push prediction higher
- Blue bars: Features that push prediction lower
- Final prediction: f(x) (shown at top)

How to read the plot:
1. Start at the base value (expected prediction)
2. Each feature adds or subtracts from the prediction
3. Features are sorted by absolute impact (largest first)
4. The cumulative effect leads to the final prediction

Key insights:
- Easily identify which features drove a specific prediction
- See the magnitude and direction of each feature's contribution
- Explain individual predictions to non-technical stakeholders
- Compare different predictions to understand model behavior

This visualization is particularly useful for:
- Model debugging and validation
- Regulatory compliance and model explainability
- Building trust with end-users
- Understanding edge cases and outliers
""")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}/")
    print("You can now:")
    print("1. View the waterfall plots to see Shapley flow for each instance")
    print("2. Compare plots across different predictions")
    print("3. Use these plots to explain model decisions to stakeholders")


if __name__ == '__main__':
    main()
