"""
Hydra-enabled experiment runner for tabular XAI models.
This script uses Hydra for configuration management, allowing easy parameter sweeps.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils.data_loader import DataLoader
from src.models import (XGBoostClassifier, LightGBMClassifier, GradientBoostingClassifier, CatBoostClassifier,
                    TabPFNClassifier, MLPClassifier, TransformerClassifier,
                    MLPDistillationClassifier, TransformerDistillationClassifier)
from src.explainability import SHAPExplainer, LIMEExplainer
from src.metrics import InterpretabilityMetrics
from src.utils.eval import Evaluator


def create_model(cfg: DictConfig, override_params: dict = None):
    """
    Create a model instance from Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        override_params: Optional dict of parameters to override the cfg.model.params
        
    Returns:
        Model instance and model type
    """
    # Support cfg.model being a plain string (e.g., when users pass overrides like model=XGBoost)
    model_node = cfg.model
    if isinstance(model_node, str):
        model_name = model_node
        model_params = {}
    else:
        model_name = getattr(model_node, 'name', str(model_node))
        model_params = OmegaConf.to_container(getattr(model_node, 'params', {}) or {}, resolve=True)

    # Apply overrides from hyperparameter tuning if provided (shallow merge)
    if override_params:
        model_params.update(override_params)
    
    # normalize model name checks
    if model_name == 'XGBoost':
        model = XGBoostClassifier(**model_params)
        model_type = 'tree'
    elif model_name == 'LightGBM':
        model = LightGBMClassifier(**model_params)
        model_type = 'tree'
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(**model_params)
        model_type = 'tree'
    elif model_name == 'CatBoost':
        print("catboost param:",model_params)
        model = CatBoostClassifier(**model_params)
        model_type = 'tree'
    elif model_name == 'TabPFN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_params['device'] = model_params.get('device', device)
        model = TabPFNClassifier(**model_params)
        model_type = 'tabpfn'
    elif model_name == 'MLP':
        model = MLPClassifier(**model_params)
        model_type = 'deep'
    elif model_name == 'Transformer':
        model = TransformerClassifier(**model_params)
        model_type = 'deep'
    elif model_name == 'MLP_Distillation':
        model = MLPDistillationClassifier(**model_params)
        model_type = 'deep'
    elif model_name == 'Transformer_Distillation':
        model = TransformerDistillationClassifier(**model_params)
        model_type = 'deep'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Return model instance, its type, and the final params dict used to instantiate it
    return model, model_type, model_params


def _canonical_model_key(model_name: str) -> str:
    """Return normalized key used in tuning summaries for a given Hydra model name."""
    name = model_name.lower()
    # common mappings
    mapping = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'gradientboosting': 'gradientboosting',
        'tabpfn': 'tabpfn',
        'mlp': 'mlp',
        'transformer': 'transformer'
    }
    return mapping.get(name, name)


def _load_best_params(results_dir: str, dataset_name: str, model_name: str, explicit_file: str = None):
    """Try to load best params for (dataset, model) from several candidate JSON files.

    Search order:
      1) explicit_file if provided
      2) results_dir/best_params.json
      3) results_dir/{dataset}_best_params.json
      4) results_dir/hyperparameter_tuning_summary.json

    Returns: dict of best params or None if not found
    """
    print("Exception file:", explicit_file)

    model_key = _canonical_model_key(model_name)


    if explicit_file and os.path.exists(explicit_file):
        try:
            with open(explicit_file, 'r') as f:
                data = json.load(f)
        except Exception:
            print(f"Warning: failed to read explicit best params file: {explicit_file}")

    print("configdata:", data)
    # If file maps dataset -> model -> info
    if dataset_name in data:
        ds_entry = data.get(dataset_name, {})
        if isinstance(ds_entry, dict) and model_key in ds_entry:
            info = ds_entry.get(model_key)
            if isinstance(info, dict) and 'best_params' in info:
                return info.get('best_params')
            if isinstance(info, dict):
                # maybe info itself is params
                return info

    # If file itself contains 'best_params' at top-level
    if isinstance(data, dict) and 'best_params' in data:
        return data.get('best_params')

    return None


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> dict:
    """
    Run experiment with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary of experiment results
    """
    print(f"\n{'='*80}")
    # Normalize model and dataset names when users pass overrides that make them plain strings
    if isinstance(cfg.model, str):
        model_name = cfg.model
    else:
        model_name = getattr(cfg.model, 'name', str(cfg.model))

    if isinstance(cfg.dataset, str):
        dataset_name = cfg.dataset
    else:
        dataset_name = getattr(cfg.dataset, 'name', str(cfg.dataset))

    print(f"Running Hydra experiment: {dataset_name} + {model_name}")
    print(f"{'='*80}\n")
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Create results directory
    results_dir = cfg.experiment.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    
    # Load data
    print("\nLoading data...")
    # When users pass dataset as a plain string (e.g., dataset=diabetes), fall back to sensible defaults
    if isinstance(cfg.dataset, str):
        ds_random_state = int(getattr(cfg, 'seed', 42))
        ds_test_size = 0.2
        ds_scale_features = True
    else:
        ds_random_state = int(cfg.dataset.random_state)
        ds_test_size = cfg.dataset.test_size
        ds_scale_features = cfg.dataset.scale_features

    loader = DataLoader(dataset_name, random_state=ds_random_state)
    X, y = loader.load_data()
    if X is None or y is None:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}'. Check data availability or DataLoader implementation.")
    
    # Save dataset to data folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save features and target
    X_path = os.path.join(data_dir, f"{dataset_name}_X.csv")
    y_path = os.path.join(data_dir, f"{dataset_name}_y.csv")
    
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False, header=['target'])
    print(f"Saved dataset to: {X_path} and {y_path}")
    
    # Number of repeated runs for averaging metrics
    repeats = cfg.experiment.get('repeats', 10)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Running {repeats} repeats to compute mean and std for test metrics")

    # Optionally load best params from tuning outputs
    # Only load/apply best-params when the user did not explicitly pass a model name via CLI.
    # This ensures we only override parameters for configured (_default) models, not ad-hoc CLI model strings.
    best_params_override = None
    explicit_best_file = cfg.experiment.get('best_params_file', None)
    print("Model name:", model_name)
    if not isinstance(cfg.model, str):
        try:
            best_params_override = _load_best_params(results_dir, dataset_name, model_name, explicit_file=explicit_best_file)
            print("New param:", best_params_override)
            if best_params_override is not None:
                print(f"Loaded best params for model {model_name} from tuning outputs; applying overrides.")
        except Exception as e:
            print(f"Warning: failed to load best params file: {e}")
    else:
        # Model was specified directly on the CLI (e.g., model=MLP). Do not auto-apply tuning overrides in this case.
        print("Model provided as CLI override; skipping automatic best-params loading.")
    
    # Initialize model
    print(f"\nInitializing {model_name}...")
    model, model_type, final_model_params = create_model(cfg, override_params=best_params_override)

    # Now that we have final_model_params, create a short config hash and experiment directory
    try:
        import json as _json
        config_hash = hash(f"{model_name}:{_json.dumps(final_model_params, sort_keys=True)}")
    except Exception:
        config_hash = hash(str(model_name))
    experiment_name = f"{dataset_name}_{model_name}_{abs(config_hash) % 10000:04d}"
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(experiment_dir, 'hydra_config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved configuration to: {config_path}")
    
    evaluator = Evaluator()
    
    # Initialize per-run metrics storage
    train_runs = []
    test_runs = []

    for i in range(repeats):
        print(f"\nRepeat {i+1}/{repeats}...")
        
        # Split data
        loader.random_state = int(ds_random_state) + i
        data = loader.prepare_data(
            X, y,
            test_size=ds_test_size,
            scale_features=ds_scale_features
        )

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train model
        model.train(X_train, y_train)

        # Evaluate metrics
        train_metrics = evaluator.evaluate_model(model, X_train, y_train)
        test_metrics = evaluator.evaluate_model(model, X_test, y_test)

        train_runs.append(train_metrics)
        test_runs.append(test_metrics)

        # print per-run results (concise)
        print("Test Metrics (run {}):".format(i+1))
        for metric, value in test_metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")

    # Aggregate test metrics across runs
    # Collect metric names
    metric_names = set()
    for tr in test_runs:
        metric_names.update([k for k, v in tr.items() if v is not None])

    test_means = {}
    test_stds = {}
    for m in metric_names:
        vals = [tr[m] for tr in test_runs if tr.get(m) is not None]
        if len(vals) > 0:
            test_means[m] = float(np.mean(vals))
            test_stds[m] = float(np.std(vals, ddof=0))
        else:
            test_means[m] = None
            test_stds[m] = None

    print("\nAggregated Test Metrics (mean ± std):")
    for m in sorted(test_means.keys()):
        mean = test_means[m]
        std = test_stds[m]
        if mean is not None:
            print(f"  {m}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {m}: None")

    # Use aggregated metrics in results
    aggregated_test_metrics = {'mean': test_means, 'std': test_stds, 'runs': test_runs}
    aggregated_train_metrics = {'runs': train_runs}

    # Generate explanations
    # print("\nGenerating explanations...")
    # try:
    #     # SHAP explanations
    #     shap_explainer = SHAPExplainer(model, X_train, model_type=model_type)
    #     # compute SHAP values (populates internal state)
    #     _ = shap_explainer.explain(X_test.head(100))

    #     # Save SHAP visualizations
    #     try:
    #         shap_explainer.plot_summary(X_test.head(100), save_path=os.path.join(experiment_dir, 'shap_summary.png'))
    #     except Exception as plot_err:
    #         print(f"Warning: SHAP summary plot failed: {plot_err}")

    #     try:
    #         shap_explainer.plot_feature_importance(X_test.head(100), save_path=os.path.join(experiment_dir, 'shap_importance.png'))
    #     except Exception as plot_err:
    #         print(f"Warning: SHAP importance plot failed: {plot_err}")

    #     # Save SHAP feature importance
    #     try:
    #         shap_importance = shap_explainer.get_feature_importance(X_test.head(100))
    #         shap_importance.to_csv(os.path.join(experiment_dir, 'shap_feature_importance.csv'), index=False)
    #     except Exception as imp_err:
    #         print(f"Warning: Could not compute or save SHAP feature importance: {imp_err}")

    #     print("✓ SHAP explanations generated")
    # except Exception as e:
    #     print(f"Warning: SHAP explanation failed: {e}")
    
    # # Try LIME if model supports it
    # if model_type in ['tree', 'deep']:
    #     try:
    #         lime_explainer = LIMEExplainer(model, X_train, X_train.columns.tolist())
    #         lime_importance = lime_explainer.get_feature_importance(X_test, num_samples=100)
    #         try:
    #             lime_explainer.plot_feature_importance(X_test, num_samples=100, save_path=os.path.join(experiment_dir, 'lime_importance.png'))
    #         except Exception as plot_err:
    #             print(f"Warning: LIME importance plot failed: {plot_err}")
    #         print("✓ LIME explanations generated")
    #     except Exception as e:
    #         print(f"Warning: LIME explanation failed: {e}")
    
    # # Calculate interpretability metrics
    # print("\nCalculating interpretability metrics...")
    # try:
    #     metrics_calc = InterpretabilityMetrics()
        
    #     # Feature importance stability (requires multiple runs)
    #     stability_score = None
        
    #     # Get feature importance from SHAP
    #     if 'shap_importance' in locals():
    #         feature_names = shap_importance['feature'].tolist()
    #         print(f"✓ Calculated interpretability metrics")
    # except Exception as e:
    #     print(f"Warning: Interpretability metrics calculation failed: {e}")
    
    # Prepare results
    results = {
        'experiment_name': experiment_name,
        'dataset': dataset_name,
        'model': model_name,
        'model_params': final_model_params,
        'best_params_loaded': best_params_override,
        'timestamp': datetime.now().isoformat(),
        'train_metrics': aggregated_train_metrics if 'aggregated_train_metrics' in locals() else train_metrics,
        'test_metrics': aggregated_test_metrics if 'aggregated_test_metrics' in locals() else test_metrics,
        'data_stats': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
    }
    
    # Save results
    results_file = os.path.join(experiment_dir, dataset_name + '_' + model_name + '_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    run_experiment()
