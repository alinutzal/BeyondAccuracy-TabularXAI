"""
Test Hydra configuration loading for gradient boosting models.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
from omegaconf import OmegaConf

def test_config_files_exist():
    """Test that all configuration files exist."""
    print("\n" + "="*80)
    print("Testing Hydra Configuration Files")
    print("="*80)
    
    config_files = [
        'conf/config.yaml',
        'conf/model/xgboost_default.yaml',
        'conf/model/xgboost_shallow.yaml',
        'conf/model/xgboost_deep.yaml',
        'conf/model/xgboost_regularized.yaml',
        'conf/model/lightgbm_default.yaml',
        'conf/model/lightgbm_shallow.yaml',
        'conf/model/lightgbm_deep.yaml',
        'conf/model/lightgbm_regularized.yaml',
        'conf/model/gradient_boosting_default.yaml',
        'conf/model/gradient_boosting_shallow.yaml',
        'conf/model/gradient_boosting_deep.yaml',
    ]
    
    print("\nChecking configuration files...")
    for config_file in config_files:
        assert os.path.exists(config_file), f"Config file not found: {config_file}"
        print(f"  ✓ {config_file}")
    
    print("\n✓ All configuration files exist!")
    return True


def test_config_loading():
    """Test that configurations can be loaded and parsed."""
    print("\n" + "="*80)
    print("Testing Configuration Loading")
    print("="*80)
    
    # Test main config
    print("\nLoading main configuration...")
    with open('conf/config.yaml', 'r') as f:
        main_config = yaml.safe_load(f)
    
    print(f"  Main config keys: {list(main_config.keys())}")
    assert 'defaults' in main_config
    assert 'dataset' in main_config
    assert 'experiment' in main_config
    print("  ✓ Main configuration loaded successfully!")
    
    # Test model configs
    model_configs = [
        'conf/model/xgboost_default.yaml',
        'conf/model/lightgbm_default.yaml',
        'conf/model/gradient_boosting_default.yaml'
    ]
    
    print("\nLoading model configurations...")
    for config_path in model_configs:
        print(f"\n  Loading {config_path}...")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        assert 'model' in cfg, f"'model' key not found in {config_path}"
        assert 'name' in cfg['model'], f"'name' not found in model config: {config_path}"
        assert 'params' in cfg['model'], f"'params' not found in model config: {config_path}"
        
        print(f"    Model name: {cfg['model']['name']}")
        print(f"    Parameters: {list(cfg['model']['params'].keys())}")
        print(f"  ✓ {config_path} loaded successfully!")
    
    print("\n✓ All configurations loaded successfully!")
    return True


def test_xgboost_configs():
    """Test XGBoost configuration variants."""
    print("\n" + "="*80)
    print("Testing XGBoost Configuration Variants")
    print("="*80)
    
    configs = {
        'default': 'conf/model/xgboost_default.yaml',
        'shallow': 'conf/model/xgboost_shallow.yaml',
        'deep': 'conf/model/xgboost_deep.yaml',
        'regularized': 'conf/model/xgboost_regularized.yaml'
    }
    
    for variant, path in configs.items():
        print(f"\n  Testing XGBoost {variant}...")
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        assert cfg['model']['name'] == 'XGBoost'
        params = cfg['model']['params']
        
        # Check required parameters
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'random_state' in params
        
        print(f"    n_estimators: {params['n_estimators']}")
        print(f"    max_depth: {params['max_depth']}")
        print(f"    learning_rate: {params['learning_rate']}")
        
        # Verify variant-specific characteristics
        if variant == 'shallow':
            assert params['max_depth'] <= 3, "Shallow config should have max_depth <= 3"
            print(f"    ✓ Verified shallow tree configuration")
        elif variant == 'deep':
            assert params['max_depth'] >= 8, "Deep config should have max_depth >= 8"
            print(f"    ✓ Verified deep tree configuration")
        elif variant == 'regularized':
            assert 'reg_alpha' in params or 'reg_lambda' in params
            print(f"    ✓ Verified regularization parameters present")
        
        print(f"  ✓ XGBoost {variant} configuration validated!")
    
    print("\n✓ All XGBoost configurations validated!")
    return True


def test_lightgbm_configs():
    """Test LightGBM configuration variants."""
    print("\n" + "="*80)
    print("Testing LightGBM Configuration Variants")
    print("="*80)
    
    configs = {
        'default': 'conf/model/lightgbm_default.yaml',
        'shallow': 'conf/model/lightgbm_shallow.yaml',
        'deep': 'conf/model/lightgbm_deep.yaml',
        'regularized': 'conf/model/lightgbm_regularized.yaml'
    }
    
    for variant, path in configs.items():
        print(f"\n  Testing LightGBM {variant}...")
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        assert cfg['model']['name'] == 'LightGBM'
        params = cfg['model']['params']
        
        # Check required parameters
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'random_state' in params
        
        print(f"    n_estimators: {params['n_estimators']}")
        print(f"    max_depth: {params['max_depth']}")
        print(f"    learning_rate: {params['learning_rate']}")
        
        # Verify variant-specific characteristics
        if variant == 'shallow':
            assert params['max_depth'] <= 3, "Shallow config should have max_depth <= 3"
            print(f"    ✓ Verified shallow tree configuration")
        elif variant == 'deep':
            assert params['max_depth'] >= 8, "Deep config should have max_depth >= 8"
            print(f"    ✓ Verified deep tree configuration")
        elif variant == 'regularized':
            assert 'reg_alpha' in params or 'reg_lambda' in params
            print(f"    ✓ Verified regularization parameters present")
        
        print(f"  ✓ LightGBM {variant} configuration validated!")
    
    print("\n✓ All LightGBM configurations validated!")
    return True


def test_gradient_boosting_configs():
    """Test GradientBoosting configuration variants."""
    print("\n" + "="*80)
    print("Testing GradientBoosting Configuration Variants")
    print("="*80)
    
    configs = {
        'default': 'conf/model/gradient_boosting_default.yaml',
        'shallow': 'conf/model/gradient_boosting_shallow.yaml',
        'deep': 'conf/model/gradient_boosting_deep.yaml'
    }
    
    for variant, path in configs.items():
        print(f"\n  Testing GradientBoosting {variant}...")
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        assert cfg['model']['name'] == 'GradientBoosting'
        params = cfg['model']['params']
        
        # Check required parameters
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'random_state' in params
        
        print(f"    n_estimators: {params['n_estimators']}")
        print(f"    max_depth: {params['max_depth']}")
        print(f"    learning_rate: {params['learning_rate']}")
        
        # Verify variant-specific characteristics
        if variant == 'shallow':
            assert params['max_depth'] <= 2, "Shallow config should have max_depth <= 2"
            print(f"    ✓ Verified shallow tree configuration")
        elif variant == 'deep':
            assert params['max_depth'] >= 5, "Deep config should have max_depth >= 5"
            print(f"    ✓ Verified deep tree configuration")
        
        print(f"  ✓ GradientBoosting {variant} configuration validated!")
    
    print("\n✓ All GradientBoosting configurations validated!")
    return True


def test_omegaconf_compatibility():
    """Test that configs work with OmegaConf."""
    print("\n" + "="*80)
    print("Testing OmegaConf Compatibility")
    print("="*80)
    
    print("\nLoading config with OmegaConf...")
    cfg = OmegaConf.load('conf/model/xgboost_default.yaml')
    
    print(f"  Config type: {type(cfg)}")
    print(f"  Model name: {cfg.model.name}")
    print(f"  Parameters: {OmegaConf.to_container(cfg.model.params)}")
    
    # Test parameter access
    assert cfg.model.name == 'XGBoost'
    assert cfg.model.params.n_estimators == 100
    assert cfg.model.params.learning_rate == 0.1
    
    print("\n✓ OmegaConf compatibility verified!")
    return True


if __name__ == '__main__':
    try:
        print("\nRunning Hydra configuration tests...")
        test_config_files_exist()
        test_config_loading()
        test_xgboost_configs()
        test_lightgbm_configs()
        test_gradient_boosting_configs()
        test_omegaconf_compatibility()
        
        print("\n" + "="*80)
        print("All Hydra configuration tests passed! ✓")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
