"""
Example demonstrating Hydra configuration for gradient boosting models.
This script shows how to use Hydra to run experiments with different parameter sets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import XGBoostClassifier, LightGBMClassifier, GradientBoostingClassifier

def main():
    """Run a simple example comparing different gradient boosting configurations."""
    
    print("\n" + "="*80)
    print("Hydra Configuration Example for Gradient Boosting Models")
    print("="*80 + "\n")
    
    print("This example demonstrates using pre-configured model parameter sets.")
    print("With Hydra, you can easily switch between configurations without code changes.\n")
    
    # Load a small dataset
    print("Loading breast cancer dataset...")
    loader = DataLoader('breast_cancer', random_state=42)
    X, y = loader.load_data()
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}\n")
    
    # Define different parameter configurations
    configurations = {
        "XGBoost Default": {
            "model_class": XGBoostClassifier,
            "params": {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        "XGBoost Shallow": {
            "model_class": XGBoostClassifier,
            "params": {
                'n_estimators': 200,
                'max_depth': 3,
                'learning_rate': 0.05,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        },
        "LightGBM Default": {
            "model_class": LightGBMClassifier,
            "params": {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1
            }
        },
        "GradientBoosting Default": {
            "model_class": GradientBoostingClassifier,
            "params": {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
    }
    
    # Train and evaluate each configuration
    results = []
    
    for config_name, config in configurations.items():
        print(f"\n{'-'*80}")
        print(f"Training: {config_name}")
        print(f"Parameters: {config['params']}")
        print(f"{'-'*80}")
        
        # Create and train model
        model = config['model_class'](**config['params'])
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        results.append({
            'Configuration': config_name,
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1_score'],
            'ROC-AUC': metrics.get('roc_auc', 'N/A')
        })
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary of Results")
    print("="*80 + "\n")
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Using Hydra for Configuration Management")
    print("="*80 + "\n")
    
    print("With Hydra, you can run these same configurations from the command line:")
    print()
    print("  # Run XGBoost with default settings")
    print("  python src/run_experiments_hydra.py model=xgboost_default")
    print()
    print("  # Run XGBoost with shallow trees (better interpretability)")
    print("  python src/run_experiments_hydra.py model=xgboost_shallow")
    print()
    print("  # Run LightGBM with deep trees")
    print("  python src/run_experiments_hydra.py model=lightgbm_deep")
    print()
    print("  # Override specific parameters")
    print("  python src/run_experiments_hydra.py model=xgboost_default model.params.n_estimators=200")
    print()
    print("  # Run parameter sweep")
    print("  python src/run_experiments_hydra.py -m model.params.learning_rate=0.01,0.05,0.1")
    print()
    print("See HYDRA_USAGE.md for more examples and documentation.")
    print()

if __name__ == '__main__':
    main()
