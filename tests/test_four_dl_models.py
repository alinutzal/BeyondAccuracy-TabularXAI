"""
Test script to verify that deep_learning is now separated into 4 models:
1. MLPClassifier (without distillation)
2. TransformerClassifier (without distillation)
3. MLPDistillationClassifier (with distillation)
4. TransformerDistillationClassifier (with distillation)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all 4 models can be imported."""
    print("\n" + "="*80)
    print("Testing imports of 4 deep learning models")
    print("="*80)
    
    try:
        from models import (MLPClassifier, TransformerClassifier, 
                          MLPDistillationClassifier, TransformerDistillationClassifier)
        print("✓ All 4 models imported successfully")
        print(f"  - MLPClassifier: {MLPClassifier}")
        print(f"  - TransformerClassifier: {TransformerClassifier}")
        print(f"  - MLPDistillationClassifier: {MLPDistillationClassifier}")
        print(f"  - TransformerDistillationClassifier: {TransformerDistillationClassifier}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_class_instantiation():
    """Test that all 4 models can be instantiated."""
    print("\n" + "="*80)
    print("Testing instantiation of 4 deep learning models")
    print("="*80)
    
    try:
        from models import (MLPClassifier, TransformerClassifier, 
                          MLPDistillationClassifier, TransformerDistillationClassifier)
        
        # Test MLPClassifier
        mlp = MLPClassifier(hidden_dims=[64, 32])
        assert mlp.distillation['enabled'] == False, "MLP should have distillation disabled by default"
        print(f"✓ MLPClassifier instantiated (distillation: {mlp.distillation['enabled']})")
        
        # Test TransformerClassifier
        transformer = TransformerClassifier(d_model=32, nhead=2)
        assert transformer.distillation['enabled'] == False, "Transformer should have distillation disabled by default"
        print(f"✓ TransformerClassifier instantiated (distillation: {transformer.distillation['enabled']})")
        
        # Test MLPDistillationClassifier
        mlp_distill = MLPDistillationClassifier(hidden_dims=[64, 32])
        assert mlp_distill.distillation['enabled'] == True, "MLPDistillation should have distillation enabled by default"
        assert mlp_distill.model_name == "MLP_Distillation", "Model name should be MLP_Distillation"
        print(f"✓ MLPDistillationClassifier instantiated (distillation: {mlp_distill.distillation['enabled']}, name: {mlp_distill.model_name})")
        
        # Test TransformerDistillationClassifier
        transformer_distill = TransformerDistillationClassifier(d_model=32, nhead=2)
        assert transformer_distill.distillation['enabled'] == True, "TransformerDistillation should have distillation enabled by default"
        assert transformer_distill.model_name == "Transformer_Distillation", "Model name should be Transformer_Distillation"
        print(f"✓ TransformerDistillationClassifier instantiated (distillation: {transformer_distill.distillation['enabled']}, name: {transformer_distill.model_name})")
        
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation_parameters():
    """Test that distillation classifiers have proper default parameters."""
    print("\n" + "="*80)
    print("Testing distillation default parameters")
    print("="*80)
    
    try:
        from models import MLPDistillationClassifier, TransformerDistillationClassifier
        
        # Test MLP distillation defaults
        mlp_distill = MLPDistillationClassifier(hidden_dims=[64, 32])
        assert mlp_distill.distillation['enabled'] == True
        assert mlp_distill.distillation['lambda'] == 0.7
        assert mlp_distill.distillation['temperature'] == 2.0
        print(f"✓ MLPDistillationClassifier defaults: lambda={mlp_distill.distillation['lambda']}, temp={mlp_distill.distillation['temperature']}")
        
        # Test Transformer distillation defaults
        transformer_distill = TransformerDistillationClassifier(d_model=32, nhead=2)
        assert transformer_distill.distillation['enabled'] == True
        assert transformer_distill.distillation['lambda'] == 0.7
        assert transformer_distill.distillation['temperature'] == 2.0
        print(f"✓ TransformerDistillationClassifier defaults: lambda={transformer_distill.distillation['lambda']}, temp={transformer_distill.distillation['temperature']}")
        
        # Test custom distillation parameters
        mlp_custom = MLPDistillationClassifier(
            hidden_dims=[64, 32],
            distillation={'lambda': 0.5, 'temperature': 3.0}
        )
        assert mlp_custom.distillation['enabled'] == True
        assert mlp_custom.distillation['lambda'] == 0.5
        assert mlp_custom.distillation['temperature'] == 3.0
        print(f"✓ MLPDistillationClassifier custom params: lambda={mlp_custom.distillation['lambda']}, temp={mlp_custom.distillation['temperature']}")
        
        return True
    except Exception as e:
        print(f"✗ Parameter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inheritance():
    """Test that distillation classifiers properly inherit from base classifiers."""
    print("\n" + "="*80)
    print("Testing inheritance relationships")
    print("="*80)
    
    try:
        from models import (MLPClassifier, TransformerClassifier, 
                          MLPDistillationClassifier, TransformerDistillationClassifier)
        
        # Test inheritance
        assert issubclass(MLPDistillationClassifier, MLPClassifier), "MLPDistillationClassifier should inherit from MLPClassifier"
        print("✓ MLPDistillationClassifier inherits from MLPClassifier")
        
        assert issubclass(TransformerDistillationClassifier, TransformerClassifier), "TransformerDistillationClassifier should inherit from TransformerClassifier"
        print("✓ TransformerDistillationClassifier inherits from TransformerClassifier")
        
        # Test instance checks
        mlp_distill = MLPDistillationClassifier(hidden_dims=[64, 32])
        assert isinstance(mlp_distill, MLPClassifier), "MLPDistillationClassifier instance should be instance of MLPClassifier"
        print("✓ MLPDistillationClassifier instance is also an MLPClassifier")
        
        transformer_distill = TransformerDistillationClassifier(d_model=32, nhead=2)
        assert isinstance(transformer_distill, TransformerClassifier), "TransformerDistillationClassifier instance should be instance of TransformerClassifier"
        print("✓ TransformerDistillationClassifier instance is also a TransformerClassifier")
        
        return True
    except Exception as e:
        print(f"✗ Inheritance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_files():
    """Test that Hydra config files exist for all 4 models."""
    print("\n" + "="*80)
    print("Testing Hydra configuration files")
    print("="*80)
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'conf', 'model')
    
    expected_configs = [
        'mlp_default.yaml',
        'transformer_default.yaml',
        'mlp_distillation.yaml',
        'transformer_distillation.yaml'
    ]
    
    all_exist = True
    for config in expected_configs:
        config_path = os.path.join(config_dir, config)
        if os.path.exists(config_path):
            print(f"✓ {config} exists")
        else:
            print(f"✗ {config} NOT FOUND")
            all_exist = False
    
    return all_exist


if __name__ == '__main__':
    try:
        print("\n" + "="*80)
        print("Testing Deep Learning Model Separation into 4 Variants")
        print("="*80)
        
        results = []
        results.append(("Imports", test_imports()))
        results.append(("Instantiation", test_class_instantiation()))
        results.append(("Distillation Parameters", test_distillation_parameters()))
        results.append(("Inheritance", test_inheritance()))
        results.append(("Config Files", test_config_files()))
        
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(passed for _, passed in results)
        if all_passed:
            print("\n" + "="*80)
            print("All tests passed! ✓")
            print("Deep learning models successfully separated into 4 variants:")
            print("  1. MLPClassifier (without distillation)")
            print("  2. TransformerClassifier (without distillation)")
            print("  3. MLPDistillationClassifier (with distillation)")
            print("  4. TransformerDistillationClassifier (with distillation)")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("Some tests failed! ✗")
            print("="*80)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
