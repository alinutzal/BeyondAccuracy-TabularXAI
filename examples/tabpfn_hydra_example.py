"""
Example demonstrating TabPFN with Hydra configuration and PyTorch Lightning support.

This example shows how to:
1. Use TabPFN with Hydra configuration management
2. Switch between different TabPFN configurations (default, fast, accurate)
3. Use the use_lightning parameter for API consistency

Note: This example requires TabPFN to be installed:
    pip install tabpfn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from omegaconf import OmegaConf
from utils.data_loader import DataLoader
from models import TabPFNClassifier


def load_config(config_name: str = 'tabpfn_default'):
    """Load a TabPFN configuration from Hydra config files."""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'conf', 
        'model', 
        f'{config_name}.yaml'
    )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    return cfg


def run_tabpfn_experiment(config_name: str = 'tabpfn_default', use_lightning: bool = False):
    """
    Run a TabPFN experiment with specified configuration.
    
    Args:
        config_name: Name of the config file (e.g., 'tabpfn_default', 'tabpfn_fast', 'tabpfn_accurate')
        use_lightning: Whether to use PyTorch Lightning wrapper
    """
    print("="*80)
    print(f"TabPFN Hydra Example - Configuration: {config_name}")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading configuration from {config_name}.yaml...")
    cfg = load_config(config_name)
    
    print("\nConfiguration:")
    print(f"  Model: {cfg.model.name}")
    print(f"  Device: {cfg.model.params.device}")
    print(f"  Ensemble configurations: {cfg.model.params.N_ensemble_configurations}")
    print(f"  Random state: {cfg.model.params.random_state}")
    print(f"  Use Lightning: {use_lightning}")
    
    # Check if TabPFN is installed
    print("\nChecking TabPFN installation...")
    try:
        model = TabPFNClassifier(use_lightning=use_lightning)
        print("✓ TabPFN is installed and ready!")
    except ImportError as e:
        print(f"✗ TabPFN is not installed: {e}")
        print("\nTo install TabPFN, run:")
        print("  pip install tabpfn")
        return 1
    
    # Load data
    print("\nLoading Breast Cancer dataset...")
    loader = DataLoader('breast_cancer', random_state=cfg.model.params.random_state)
    X, y = loader.load_data()
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Create model with config parameters
    print(f"\nCreating TabPFN model with {config_name} configuration...")
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model = TabPFNClassifier(use_lightning=use_lightning, **model_params)
    
    # Train model
    print("\nTraining TabPFN model (in-context learning)...")
    model.train(X_train, y_train)
    print("✓ Model trained successfully!")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")
    
    print("\n" + "="*80)
    print(f"✓ Experiment completed successfully with {config_name}!")
    print("="*80)
    
    return 0


def compare_configurations():
    """Compare different TabPFN configurations."""
    print("\n" + "="*80)
    print("Comparing TabPFN Configurations")
    print("="*80)
    
    configs = [
        ('tabpfn_fast', 'Fast - 8 ensembles'),
        ('tabpfn_default', 'Default - 32 ensembles'),
        ('tabpfn_accurate', 'Accurate - 64 ensembles')
    ]
    
    print("\nAvailable configurations:")
    for config_name, description in configs:
        cfg = load_config(config_name)
        n_ensemble = cfg.model.params.N_ensemble_configurations
        print(f"  {config_name:20s}: {n_ensemble:3d} ensembles - {description}")
    
    print("\nTo run with a specific configuration:")
    print("  python tabpfn_hydra_example.py <config_name>")
    print("\nExamples:")
    print("  python tabpfn_hydra_example.py tabpfn_fast")
    print("  python tabpfn_hydra_example.py tabpfn_default")
    print("  python tabpfn_hydra_example.py tabpfn_accurate")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TabPFN Hydra Configuration Example',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='tabpfn_default',
        choices=['tabpfn_default', 'tabpfn_fast', 'tabpfn_accurate'],
        help='Configuration to use (default: tabpfn_default)'
    )
    parser.add_argument(
        '--use-lightning',
        action='store_true',
        help='Use PyTorch Lightning wrapper (for API consistency)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show comparison of all configurations'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations()
        return 0
    
    return run_tabpfn_experiment(args.config, args.use_lightning)


if __name__ == '__main__':
    sys.exit(main())
