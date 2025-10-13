"""
Simple test to verify ShapIQ module structure without requiring all dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_module_exists():
    """Test that the shapiq_explainer module exists."""
    print("Testing module existence...")
    
    try:
        import explainability.shapiq_explainer
        print("✓ shapiq_explainer module exists")
        return True
    except Exception as e:
        print(f"✗ Failed to find module: {e}")
        return False


def test_class_defined():
    """Test that ShapIQExplainer class is defined."""
    print("\nTesting class definition...")
    
    try:
        from explainability.shapiq_explainer import ShapIQExplainer
        print("✓ ShapIQExplainer class is defined")
        print(f"  - Class name: {ShapIQExplainer.__name__}")
        print(f"  - Module: {ShapIQExplainer.__module__}")
        return True
    except Exception as e:
        print(f"✗ Failed to import class: {e}")
        return False


def test_class_methods():
    """Test that ShapIQExplainer has expected methods."""
    print("\nTesting class methods...")
    
    try:
        from explainability.shapiq_explainer import ShapIQExplainer
        
        expected_methods = [
            '__init__',
            'explain',
            'explain_instance',
            'get_interaction_strength',
            'plot_interaction_network',
            'plot_interaction_heatmap',
        ]
        
        missing_methods = []
        for method in expected_methods:
            if not hasattr(ShapIQExplainer, method):
                missing_methods.append(method)
            else:
                print(f"  ✓ Method '{method}' exists")
        
        if missing_methods:
            print(f"✗ Missing methods: {missing_methods}")
            return False
        
        print("✓ All expected methods are defined")
        return True
        
    except Exception as e:
        print(f"✗ Failed to check methods: {e}")
        return False


def test_docstrings():
    """Test that key methods have docstrings."""
    print("\nTesting docstrings...")
    
    try:
        from explainability.shapiq_explainer import ShapIQExplainer
        
        methods_to_check = ['explain', 'explain_instance', 'get_interaction_strength']
        
        for method_name in methods_to_check:
            method = getattr(ShapIQExplainer, method_name)
            if method.__doc__:
                print(f"  ✓ Method '{method_name}' has docstring")
            else:
                print(f"  ⚠ Method '{method_name}' missing docstring")
        
        print("✓ Docstring check completed")
        return True
        
    except Exception as e:
        print(f"✗ Failed to check docstrings: {e}")
        return False


def test_export():
    """Test that ShapIQExplainer is exported from explainability package."""
    print("\nTesting package exports...")
    
    try:
        # Check if it's in __all__
        import explainability
        if hasattr(explainability, '__all__'):
            if 'ShapIQExplainer' in explainability.__all__:
                print("  ✓ ShapIQExplainer in __all__")
            else:
                print("  ✗ ShapIQExplainer not in __all__")
                return False
        
        # Check if it can be imported directly
        from explainability import ShapIQExplainer
        print("  ✓ Can import ShapIQExplainer from explainability")
        
        print("✓ Export check passed")
        return True
        
    except Exception as e:
        print(f"✗ Failed export check: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements():
    """Test that shapiq is in requirements.txt."""
    print("\nTesting requirements.txt...")
    
    try:
        req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        with open(req_path, 'r') as f:
            content = f.read()
        
        if 'shapiq' in content:
            print("  ✓ shapiq found in requirements.txt")
            return True
        else:
            print("  ✗ shapiq not found in requirements.txt")
            return False
            
    except Exception as e:
        print(f"✗ Failed to check requirements.txt: {e}")
        return False


def main():
    """Run all simple tests."""
    print("="*80)
    print("ShapIQ Simple Integration Tests")
    print("(Testing module structure without dependencies)")
    print("="*80)
    
    tests = [
        test_module_exists,
        test_class_defined,
        test_class_methods,
        test_docstrings,
        test_export,
        test_requirements,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All structure tests passed!")
        print("\nNote: Full functional tests require dependencies:")
        print("  - shapiq, xgboost, shap, networkx")
        print("  Run: pip install shapiq xgboost shap networkx")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
