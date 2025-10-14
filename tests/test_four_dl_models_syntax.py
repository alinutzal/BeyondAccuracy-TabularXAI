"""
Test script to verify syntax and structure of the 4 deep learning models.
This test doesn't require numpy or other dependencies.
"""
import sys
import os
import ast

def test_syntax():
    """Test that deep_learning.py has valid Python syntax."""
    print("\n" + "="*80)
    print("Testing Python syntax")
    print("="*80)
    
    deep_learning_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'models', 'deep_learning.py'
    )
    
    try:
        with open(deep_learning_path, 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print("✓ deep_learning.py has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in deep_learning.py: {e}")
        return False


def test_class_definitions():
    """Test that all 4 classes are defined in deep_learning.py."""
    print("\n" + "="*80)
    print("Testing class definitions")
    print("="*80)
    
    deep_learning_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'models', 'deep_learning.py'
    )
    
    try:
        with open(deep_learning_path, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        expected_classes = ['MLPClassifier', 'TransformerClassifier', 
                          'MLPDistillationClassifier', 'TransformerDistillationClassifier']
        
        all_found = True
        for cls_name in expected_classes:
            if cls_name in classes:
                print(f"✓ {cls_name} is defined")
            else:
                print(f"✗ {cls_name} NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Failed to check class definitions: {e}")
        return False


def test_inheritance():
    """Test that distillation classes inherit from base classes."""
    print("\n" + "="*80)
    print("Testing inheritance (AST analysis)")
    print("="*80)
    
    deep_learning_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'models', 'deep_learning.py'
    )
    
    try:
        with open(deep_learning_path, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Find class definitions and their bases
        class_bases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                class_bases[node.name] = bases
        
        # Check MLPDistillationClassifier inherits from MLPClassifier
        if 'MLPDistillationClassifier' in class_bases:
            if 'MLPClassifier' in class_bases['MLPDistillationClassifier']:
                print("✓ MLPDistillationClassifier inherits from MLPClassifier")
            else:
                print(f"✗ MLPDistillationClassifier bases: {class_bases['MLPDistillationClassifier']}")
                return False
        else:
            print("✗ MLPDistillationClassifier not found")
            return False
        
        # Check TransformerDistillationClassifier inherits from TransformerClassifier
        if 'TransformerDistillationClassifier' in class_bases:
            if 'TransformerClassifier' in class_bases['TransformerDistillationClassifier']:
                print("✓ TransformerDistillationClassifier inherits from TransformerClassifier")
            else:
                print(f"✗ TransformerDistillationClassifier bases: {class_bases['TransformerDistillationClassifier']}")
                return False
        else:
            print("✗ TransformerDistillationClassifier not found")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to check inheritance: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_init_exports():
    """Test that __init__.py exports all 4 models."""
    print("\n" + "="*80)
    print("Testing __init__.py exports")
    print("="*80)
    
    init_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'models', '__init__.py'
    )
    
    try:
        with open(init_path, 'r') as f:
            code = f.read()
        
        expected_exports = ['MLPClassifier', 'TransformerClassifier', 
                          'MLPDistillationClassifier', 'TransformerDistillationClassifier']
        
        all_exported = True
        for export in expected_exports:
            if export in code:
                print(f"✓ {export} is exported")
            else:
                print(f"✗ {export} NOT EXPORTED")
                all_exported = False
        
        return all_exported
    except Exception as e:
        print(f"✗ Failed to check exports: {e}")
        return False


def test_run_experiments_hydra():
    """Test that run_experiments_hydra.py supports all 4 models."""
    print("\n" + "="*80)
    print("Testing run_experiments_hydra.py")
    print("="*80)
    
    hydra_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'run_experiments_hydra.py'
    )
    
    try:
        with open(hydra_path, 'r') as f:
            code = f.read()
        
        # Check imports
        if 'MLPDistillationClassifier' in code and 'TransformerDistillationClassifier' in code:
            print("✓ New classes are imported")
        else:
            print("✗ New classes are NOT imported")
            return False
        
        # Check model creation logic
        if "'MLP_Distillation'" in code and "'Transformer_Distillation'" in code:
            print("✓ Model creation logic includes new model types")
        else:
            print("✗ Model creation logic doesn't include new model types")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to check run_experiments_hydra.py: {e}")
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


def test_config_content():
    """Test that config files have correct model names."""
    print("\n" + "="*80)
    print("Testing Hydra configuration content")
    print("="*80)
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'conf', 'model')
    
    configs = [
        ('mlp_distillation.yaml', 'MLP_Distillation'),
        ('transformer_distillation.yaml', 'Transformer_Distillation')
    ]
    
    all_correct = True
    for config_file, expected_name in configs:
        config_path = os.path.join(config_dir, config_file)
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            if f'name: {expected_name}' in content:
                print(f"✓ {config_file} has correct model name: {expected_name}")
            else:
                print(f"✗ {config_file} doesn't have correct model name")
                all_correct = False
            
            if 'distillation:' in content and 'enabled: true' in content:
                print(f"✓ {config_file} has distillation enabled")
            else:
                print(f"✗ {config_file} doesn't have distillation enabled")
                all_correct = False
        except Exception as e:
            print(f"✗ Failed to read {config_file}: {e}")
            all_correct = False
    
    return all_correct


if __name__ == '__main__':
    try:
        print("\n" + "="*80)
        print("Testing Deep Learning Model Separation - Syntax & Structure")
        print("="*80)
        
        results = []
        results.append(("Syntax", test_syntax()))
        results.append(("Class Definitions", test_class_definitions()))
        results.append(("Inheritance", test_inheritance()))
        results.append(("__init__.py Exports", test_init_exports()))
        results.append(("run_experiments_hydra.py", test_run_experiments_hydra()))
        results.append(("Config Files Exist", test_config_files()))
        results.append(("Config File Content", test_config_content()))
        
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(passed for _, passed in results)
        if all_passed:
            print("\n" + "="*80)
            print("All syntax and structure tests passed! ✓")
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
