"""
Main script to run experiments across datasets and models.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import DataLoader
from models import XGBoostClassifier, LightGBMClassifier, MLPClassifier, TransformerClassifier
from explainability import SHAPExplainer, LIMEExplainer
from metrics import InterpretabilityMetrics


def experiment_exists(dataset_name: str, model_name: str, results_dir: str) -> bool:
    """
    Check if experiment results already exist for a dataset and model combination.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        results_dir: Directory where results are stored
    
    Returns:
        True if results.json exists in the experiment directory, False otherwise
    """
    experiment_dir = os.path.join(results_dir, f"{dataset_name}_{model_name}")
    results_file = os.path.join(experiment_dir, 'results.json')
    return os.path.exists(results_file)


def run_experiment(dataset_name: str, model_name: str, results_dir: str = '../results', rerun: bool = False):
    """
    Run experiment for a specific dataset and model combination.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        results_dir: Directory to save results
        rerun: If False and results already exist, skip the experiment. If True, rerun regardless.
    """
    # Check if experiment results already exist
    if not rerun and experiment_exists(dataset_name, model_name, results_dir):
        experiment_dir = os.path.join(results_dir, f"{dataset_name}_{model_name}")
        print(f"\n{'='*80}")
        print(f"Skipping experiment: {dataset_name} + {model_name}")
        print(f"Results already exist at: {experiment_dir}")
        print(f"Use --rerun flag to force rerun this experiment")
        print(f"{'='*80}\n")
        
        # Load and return existing results
        results_file = os.path.join(experiment_dir, 'results.json')
        with open(results_file, 'r') as f:
            return json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {dataset_name} + {model_name}")
    print(f"{'='*80}\n")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    experiment_dir = os.path.join(results_dir, f"{dataset_name}_{model_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    loader = DataLoader(dataset_name, random_state=42)
    X, y = loader.load_data()
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize model
    print(f"\nInitializing {model_name}...")
    if model_name == 'XGBoost':
        model = XGBoostClassifier(n_estimators=100, max_depth=6, random_state=42)
        model_type = 'tree'
    elif model_name == 'LightGBM':
        model = LightGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
        model_type = 'tree'
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_dims=[128, 64, 32], epochs=50, batch_size=32)
        model_type = 'deep'
    elif model_name == 'Transformer':
        model = TransformerClassifier(d_model=64, nhead=4, num_layers=2, epochs=50, batch_size=32)
        model_type = 'deep'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model
    print(f"Training {model_name}...")
    model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}" if value is not None else f"  {metric}: N/A")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}" if value is not None else f"  {metric}: N/A")
    
    # SHAP explanations
    print("\nGenerating SHAP explanations...")
    try:
        shap_explainer = SHAPExplainer(model, X_train, model_type=model_type)
        shap_values = shap_explainer.explain(X_test.head(100))
        shap_importance = shap_explainer.get_feature_importance(X_test.head(100))
        
        # Save SHAP plots
        try:
            shap_explainer.plot_summary(X_test.head(100), 
                                        save_path=os.path.join(experiment_dir, 'shap_summary.png'))
        except Exception as plot_err:
            print(f"Warning: SHAP summary plot failed: {plot_err}")
        
        try:
            shap_explainer.plot_feature_importance(X_test.head(100),
                                                   save_path=os.path.join(experiment_dir, 'shap_importance.png'))
        except Exception as plot_err:
            print(f"Warning: SHAP importance plot failed: {plot_err}")
        
        # Save SHAP dependence plots for top features
        try:
            # Get top 3 most important features
            top_features = shap_importance.head(3)['feature'].tolist()
            for feature in top_features:
                safe_feature_name = feature.replace(' ', '_').replace('/', '_')
                shap_explainer.plot_dependence(
                    feature,
                    X_test.head(100),
                    save_path=os.path.join(experiment_dir, f'shap_dependence_{safe_feature_name}.png')
                )
            print(f"Generated SHAP dependence plots for top {len(top_features)} features")
        except Exception as plot_err:
            print(f"Warning: SHAP dependence plot failed: {plot_err}")
        
        # Get SHAP explanations for metrics
        shap_explanations = []
        for i in range(min(50, len(X_test))):
            try:
                exp = shap_explainer.explain_instance(X_test.iloc[i])
                shap_explanations.append(exp)
            except Exception as exp_err:
                print(f"Warning: SHAP explanation for instance {i} failed: {exp_err}")
                continue
        
        print(f"Generated SHAP explanations for {len(shap_explanations)} instances")
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        import traceback
        traceback.print_exc()
        shap_importance = None
        shap_explanations = []
    
    # LIME explanations
    print("\nGenerating LIME explanations...")
    try:
        lime_explainer = LIMEExplainer(model, X_train, X_train.columns.tolist())
        lime_importance = lime_explainer.get_feature_importance(X_test, num_samples=50)
        
        # Save LIME plots
        try:
            lime_explainer.plot_feature_importance(X_test, num_samples=50,
                                                  save_path=os.path.join(experiment_dir, 'lime_importance.png'))
        except Exception as plot_err:
            print(f"Warning: LIME plot failed: {plot_err}")
        
        # Get LIME explanations for metrics
        lime_explanations = []
        for i in range(min(50, len(X_test))):
            try:
                exp = lime_explainer.explain_instance(X_test.iloc[i].values, num_features=10)
                try:
                    exp_dict = dict(exp.as_list())
                except KeyError:
                    if hasattr(exp, 'local_exp') and len(exp.local_exp) > 0:
                        first_label = next(iter(exp.local_exp.keys()))
                        exp_dict = dict(exp.as_list(label=first_label))
                    else:
                        exp_dict = {}
                # Convert to feature-based dict
                feature_exp = {}
                for feature in X_test.columns:
                    for key, val in exp_dict.items():
                        if feature in key:
                            feature_exp[feature] = val
                            break
                    if feature not in feature_exp:
                        feature_exp[feature] = 0.0
                lime_explanations.append(feature_exp)
            except Exception as exp_err:
                print(f"Warning: LIME explanation for instance {i} failed: {exp_err}")
                continue
        
        print(f"Generated LIME explanations for {len(lime_explanations)} instances")
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        import traceback
        traceback.print_exc()
        lime_importance = None
        lime_explanations = []
    
    # Calculate interpretability metrics
    print("\nCalculating interpretability metrics...")
    interpretability_metrics = {}
    
    try:
        # Collect multiple importance runs for stability (using SHAP)
        importance_runs = []
        if shap_importance is not None:
            importance_runs.append(shap_importance)
            # Run SHAP on different subsets for stability testing
            for seed in [42, 43, 44]:
                subset_indices = np.random.RandomState(seed).choice(len(X_test), size=min(50, len(X_test)), replace=False)
                X_subset = X_test.iloc[subset_indices]
                shap_explainer_subset = SHAPExplainer(model, X_train, model_type=model_type)
                _ = shap_explainer_subset.explain(X_subset)
                importance_subset = shap_explainer_subset.get_feature_importance(X_subset)
                importance_runs.append(importance_subset)
        
        if lime_importance is not None and len(importance_runs) > 0:
            importance_runs.append(lime_importance)
        
        # Compute all interpretability metrics
        if len(importance_runs) > 1:
            interpretability_metrics = InterpretabilityMetrics.compute_all_metrics(
                model, X_test, importance_runs, shap_explanations, lime_explanations
            )
            
            print("\nInterpretability Metrics:")
            for metric, value in interpretability_metrics.items():
                print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error calculating interpretability metrics: {e}")
    
    # Save results - convert numpy types to native Python types
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'dataset_info': convert_to_native(loader.get_dataset_info()),
        'train_metrics': convert_to_native(train_metrics),
        'test_metrics': convert_to_native(test_metrics),
        'interpretability_metrics': convert_to_native(interpretability_metrics)
    }
    
    # Save as JSON
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature importance
    if shap_importance is not None:
        shap_importance.to_csv(os.path.join(experiment_dir, 'shap_feature_importance.csv'), index=False)
    if lime_importance is not None:
        lime_importance.to_csv(os.path.join(experiment_dir, 'lime_feature_importance.csv'), index=False)
    
    print(f"\nResults saved to {experiment_dir}")
    
    return results


def run_all_experiments(results_dir: str = '../results', rerun: bool = False):
    """
    Run experiments for all dataset and model combinations.
    
    Args:
        results_dir: Directory to save results
        rerun: If False and results already exist, skip the experiment. If True, rerun regardless.
    """
    datasets = ['breast_cancer', 'adult_income', 'bank_marketing']
    models = ['XGBoost', 'LightGBM', 'Transformer']
    
    all_results = []
    
    for dataset in datasets:
        for model in models:
            try:
                result = run_experiment(dataset, model, results_dir, rerun=rerun)
                all_results.append(result)
            except Exception as e:
                print(f"\nError running {dataset} + {model}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save aggregated results
    summary = []
    for result in all_results:
        summary_row = {
            'dataset': result['dataset'],
            'model': result['model'],
            **{f'test_{k}': v for k, v in result['test_metrics'].items()},
            **result['interpretability_metrics']
        }
        summary.append(summary_row)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"\nSummary saved to {os.path.join(results_dir, 'summary.csv')}")
    
    return summary_df


if __name__ == '__main__':
    import sys
    
    # Get absolute path to results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), 'results')
    
    # Check for --rerun flag
    rerun = '--rerun' in sys.argv
    if rerun:
        sys.argv.remove('--rerun')
    
    if len(sys.argv) > 1:
        # Run single experiment
        dataset = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else 'XGBoost'
        run_experiment(dataset, model, results_dir, rerun=rerun)
    else:
        # Run all experiments
        summary = run_all_experiments(results_dir, rerun=rerun)
        print("\n" + summary.to_string())
