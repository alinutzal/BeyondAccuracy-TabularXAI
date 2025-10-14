"""
Example demonstrating how to use Hydra configurations for deep learning models.
This script shows how to programmatically load and use Hydra configurations
for MLP and Transformer models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

def main():
    print("\n" + "="*80)
    print("Deep Learning Models with Hydra - Example")
    print("="*80 + "\n")
    
    # Get the configuration directory path
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'conf')
    config_dir = os.path.abspath(config_dir)
    
    print(f"Configuration directory: {config_dir}\n")
    
    # Example 1: Load MLP default configuration
    print("="*80)
    print("Example 1: Loading MLP Default Configuration")
    print("="*80)
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name='config', overrides=['model=mlp_default'])
        
        print(f"\nModel: {cfg.model.name}")
        print(f"Hidden dimensions: {cfg.model.params.hidden_dims}")
        print(f"Activation: {cfg.model.params.activation}")
        print(f"Dropout: {cfg.model.params.dropout}")
        print(f"Batch size: {cfg.model.params.training.batch_size}")
        print(f"Epochs: {cfg.model.params.training.epochs}")
        print(f"Learning rate: {cfg.model.params.optimizer.lr}")
    
    # Example 2: Load Transformer configuration
    print("\n" + "="*80)
    print("Example 2: Loading Transformer Default Configuration")
    print("="*80)
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name='config', overrides=['model=transformer_default'])
        
        print(f"\nModel: {cfg.model.name}")
        print(f"d_model: {cfg.model.params.d_model}")
        print(f"Number of heads: {cfg.model.params.nhead}")
        print(f"Number of layers: {cfg.model.params.num_layers}")
        print(f"Dropout: {cfg.model.params.dropout}")
        print(f"Batch size: {cfg.model.params.training.batch_size}")
        print(f"Epochs: {cfg.model.params.training.epochs}")
    
    # Example 3: Override parameters
    print("\n" + "="*80)
    print("Example 3: Overriding Parameters")
    print("="*80)
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name='config',
            overrides=[
                'model=mlp_default',
                'model.params.hidden_dims=[256,128,64]',
                'model.params.optimizer.lr=0.01',
                'model.params.training.batch_size=256'
            ]
        )
        
        print(f"\nModel: {cfg.model.name}")
        print(f"Hidden dimensions (overridden): {cfg.model.params.hidden_dims}")
        print(f"Learning rate (overridden): {cfg.model.params.optimizer.lr}")
        print(f"Batch size (overridden): {cfg.model.params.training.batch_size}")
    
    # Example 4: Compare all MLP variants
    print("\n" + "="*80)
    print("Example 4: Comparing MLP Variants")
    print("="*80 + "\n")
    
    mlp_variants = ['mlp_default', 'mlp_small', 'mlp_large']
    
    for variant in mlp_variants:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name='config', overrides=[f'model={variant}'])
            
            print(f"{variant}:")
            print(f"  Hidden dims: {cfg.model.params.hidden_dims}")
            print(f"  Activation: {cfg.model.params.activation}")
            print(f"  Batch size: {cfg.model.params.training.batch_size}")
            print(f"  Epochs: {cfg.model.params.training.epochs}")
            print(f"  MixUp enabled: {cfg.model.params.mixup.enabled}")
            print()
    
    # Example 5: Compare all Transformer variants
    print("="*80)
    print("Example 5: Comparing Transformer Variants")
    print("="*80 + "\n")
    
    transformer_variants = ['transformer_default', 'transformer_small', 'transformer_large']
    
    for variant in transformer_variants:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name='config', overrides=[f'model={variant}'])
            
            print(f"{variant}:")
            print(f"  d_model: {cfg.model.params.d_model}")
            print(f"  heads: {cfg.model.params.nhead}")
            print(f"  layers: {cfg.model.params.num_layers}")
            print(f"  Batch size: {cfg.model.params.training.batch_size}")
            print(f"  Epochs: {cfg.model.params.training.epochs}")
            print(f"  MixUp enabled: {cfg.model.params.mixup.enabled}")
            print()
    
    print("="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
    
    print("To run experiments with these configurations, use:")
    print("  cd src")
    print("  python run_experiments_hydra.py model=mlp_default")
    print("  python run_experiments_hydra.py model=transformer_default")
    print()

if __name__ == '__main__':
    main()
