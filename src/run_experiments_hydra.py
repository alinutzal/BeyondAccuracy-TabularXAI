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

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from utils.data_loader import DataLoader
from models import (XGBoostClassifier, LightGBMClassifier, GradientBoostingClassifier, CatBoostClassifier,
                    TabPFNClassifier, MLPClassifier, TransformerClassifier,
                    MLPDistillationClassifier, TransformerDistillationClassifier)
from explainability import SHAPExplainer, LIMEExplainer
from metrics import InterpretabilityMetrics
from utils.eval import Evaluator


def create_model(cfg: DictConfig, override_params: dict = None):
    """
    Create a model instance from Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        override_params: Optional dict of parameters to override the cfg.model.params
        
    Returns:
        Model instance and model type
    """
    model_name = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)

    # Apply overrides from hyperparameter tuning if provided
    if override_params:
        # shallow merge - override or add keys from override_params
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
    
    return model, model_type


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
    candidates = []
    if explicit_file:
        candidates.append(explicit_file)
    candidates.append(os.path.join(results_dir, 'best_params.json'))
    candidates.append(os.path.join(results_dir, f"{dataset_name}_best_params.json"))
    candidates.append(os.path.join(results_dir, 'hyperparameter_tuning_summary.json'))

    model_key = _canonical_model_key(model_name)

    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            # If file is per-dataset (top-level keys are model keys)
            if model_key in data:
                entry = data.get(model_key)
                # entry might be {best_score, best_params} or directly params
                if isinstance(entry, dict) and 'best_params' in entry:
                    return entry.get('best_params')
                if isinstance(entry, dict) and any(k in entry for k in ['best_score', 'best_params']):
                    return entry.get('best_params')
                # otherwise assume entry itself is params dict
                if isinstance(entry, dict):
                    return entry

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
    print(f"Running Hydra experiment: {cfg.dataset.name} + {cfg.model.name}")
    print(f"{'='*80}\n")
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Create results directory
    results_dir = cfg.experiment.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Get config hash or use default experiment name
    config_hash = hash(OmegaConf.to_yaml(cfg.model))
    experiment_name = f"{cfg.dataset.name}_{cfg.model.name}_{abs(config_hash) % 10000:04d}"
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, 'hydra_config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved configuration to: {config_path}")
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader(cfg.dataset.name, random_state=cfg.dataset.random_state)
    X, y = loader.load_data()
    
    # Number of repeated runs for averaging metrics
    repeats = cfg.experiment.get('repeats', 10)
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Optionally load best params from tuning outputs
    best_params_override = None
    explicit_best_file = cfg.experiment.get('best_params_file', None)
    try:
        best_params_override = _load_best_params(results_dir, cfg.dataset.name, cfg.model.name, explicit_file=explicit_best_file)
        if best_params_override is not None:
            print(f"Loaded best params for model {cfg.model.name} from tuning outputs; applying overrides.")
    except Exception as e:
        print(f"Warning: failed to load best params file: {e}")
    
    # Initialize model
    print(f"\nInitializing {cfg.model.name}...")
    model, model_type = create_model(cfg, override_params=best_params_override)
    
    evaluator = Evaluator()
    
    # Initialize per-run metrics storage
    train_runs = []
    test_runs = []

    for i in range(repeats):
        print(f"\nRepeat {i+1}/{repeats}...")
        
        # Split data
        loader.random_state = int(cfg.dataset.random_state) + i
        data = loader.prepare_data(
            X, y,
            test_size=cfg.dataset.test_size,
            scale_features=cfg.dataset.scale_features
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
    print("\nGenerating explanations...")
    try:
        # SHAP explanations
        shap_explainer = SHAPExplainer(model, X_train, model_type=model_type)
        shap_values = shap_explainer.explain(X_test.head(100))
        
        # Save SHAP visualizations
        shap_explainer.plot_summary(shap_values, X_test.head(100), save_path=experiment_dir)
        shap_explainer.plot_importance(shap_values, X_test.head(100), save_path=experiment_dir)
        
        # Save SHAP feature importance
        shap_importance = shap_explainer.get_feature_importance(shap_values, X_test.head(100))
        shap_importance.to_csv(os.path.join(experiment_dir, 'shap_feature_importance.csv'), index=False)
        
        print("✓ SHAP explanations generated")
    except Exception as e:
        print(f"Warning: SHAP explanation failed: {e}")
    
    # Try LIME if model supports it
    if model_type in ['tree', 'deep']:
        try:
            lime_explainer = LIMEExplainer(model, X_train, mode='classification')
            lime_importance = lime_explainer.explain_instance(X_test.iloc[0])
            lime_explainer.plot_importance(lime_importance, save_path=experiment_dir)
            print("✓ LIME explanations generated")
        except Exception as e:
            print(f"Warning: LIME explanation failed: {e}")
    
    # Calculate interpretability metrics
    print("\nCalculating interpretability metrics...")
    try:
        metrics_calc = InterpretabilityMetrics()
        
        # Feature importance stability (requires multiple runs)
        stability_score = None
        
        # Get feature importance from SHAP
        if 'shap_importance' in locals():
            feature_names = shap_importance['feature'].tolist()
            print(f"✓ Calculated interpretability metrics")
    except Exception as e:
        print(f"Warning: Interpretability metrics calculation failed: {e}")
    
    # Prepare results
    results = {
        'experiment_name': experiment_name,
        'dataset': cfg.dataset.name,
        'model': cfg.model.name,
        'model_params': OmegaConf.to_container(cfg.model.params, resolve=True),
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
    results_file = os.path.join(experiment_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    run_experiment()
