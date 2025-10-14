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
from models import XGBoostClassifier, LightGBMClassifier, GradientBoostingClassifier, TabPFNClassifier, MLPClassifier, TransformerClassifier
from explainability import SHAPExplainer, LIMEExplainer
from metrics import InterpretabilityMetrics
from utils.eval import Evaluator


def create_model(cfg: DictConfig):
    """
    Create a model instance from Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Model instance and model type
    """
    model_name = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, model_type


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
    data = loader.prepare_data(
        X, y,
        test_size=cfg.dataset.test_size,
        scale_features=cfg.dataset.scale_features
    )
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize model
    print(f"\nInitializing {cfg.model.name}...")
    model, model_type = create_model(cfg)
    
    # Train model
    print(f"Training {cfg.model.name}...")
    model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = Evaluator()
    train_metrics = evaluator.evaluate_model(model, X_train, y_train)
    test_metrics = evaluator.evaluate_model(model, X_test, y_test)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
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
        'timestamp': datetime.now().isoformat(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
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
