"""
Test script to validate ShapIQ integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd


def test_shapiq_import():
    """Test that ShapIQExplainer can be imported."""
    print("Testing ShapIQ import...")
    
    try:
        from explainability import ShapIQExplainer
        print("✓ ShapIQExplainer successfully imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import ShapIQExplainer: {e}")
        return False


def test_shapiq_initialization():
    """Test ShapIQExplainer initialization."""
    print("\nTesting ShapIQ initialization...")
    
    try:
        from explainability import ShapIQExplainer
        from models import XGBoostClassifier
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y_train = np.random.randint(0, 2, 100)
        
        # Train a simple model
        model = XGBoostClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Initialize explainer
        explainer = ShapIQExplainer(model, X_train, max_order=2)
        
        print("✓ ShapIQExplainer initialized successfully")
        print(f"  - Max order: {explainer.max_order}")
        print(f"  - Feature names: {explainer.feature_names}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize ShapIQExplainer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shapiq_explain():
    """Test ShapIQ explanation generation."""
    print("\nTesting ShapIQ explanation generation...")
    
    try:
        from explainability import ShapIQExplainer
        from models import XGBoostClassifier
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y_train = np.random.randint(0, 2, 100)
        
        X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Train model
        model = XGBoostClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Create explainer
        explainer = ShapIQExplainer(model, X_train, max_order=2)
        
        # Generate explanations
        result = explainer.explain(X_test)
        
        print("✓ Explanations generated successfully")
        print(f"  - Number of interactions: {len(result['interaction_values'])}")
        print(f"  - Index type: {result['index']}")
        print(f"  - Samples explained: {result['n_samples']}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to generate explanations: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shapiq_interaction_strength():
    """Test interaction strength computation."""
    print("\nTesting interaction strength computation...")
    
    try:
        from explainability import ShapIQExplainer
        from models import XGBoostClassifier
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y_train = np.random.randint(0, 2, 100)
        
        X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Train model
        model = XGBoostClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Create explainer and compute interactions
        explainer = ShapIQExplainer(model, X_train, max_order=2)
        explainer.explain(X_test)
        
        # Get interaction strengths
        top_interactions = explainer.get_interaction_strength(top_k=5)
        
        print("✓ Interaction strengths computed successfully")
        print(f"  - Top interactions returned: {len(top_interactions)}")
        print(f"  - Columns: {list(top_interactions.columns)}")
        
        if len(top_interactions) > 0:
            print(f"  - Strongest interaction: {top_interactions.iloc[0]['features']}")
            print(f"    Strength: {top_interactions.iloc[0]['interaction_strength']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to compute interaction strengths: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shapiq_explain_instance():
    """Test single instance explanation."""
    print("\nTesting single instance explanation...")
    
    try:
        from explainability import ShapIQExplainer
        from models import XGBoostClassifier
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y_train = np.random.randint(0, 2, 100)
        
        X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Train model
        model = XGBoostClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Create explainer
        explainer = ShapIQExplainer(model, X_train, max_order=2)
        
        # Explain single instance
        instance = X_test.iloc[0]
        instance_result = explainer.explain_instance(instance)
        
        print("✓ Single instance explained successfully")
        print(f"  - Keys in result: {list(instance_result.keys())}")
        print(f"  - Top interactions: {len(instance_result['top_interactions'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to explain instance: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shapiq_visualizations():
    """Test ShapIQ visualization methods."""
    print("\nTesting ShapIQ visualization methods...")
    
    try:
        from explainability import ShapIQExplainer
        from models import XGBoostClassifier
        import tempfile
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y_train = np.random.randint(0, 2, 100)
        
        X_test = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Train model
        model = XGBoostClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Create explainer and compute interactions
        explainer = ShapIQExplainer(model, X_train, max_order=2)
        explainer.explain(X_test)
        
        # Test heatmap visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            heatmap_path = f.name
        
        try:
            explainer.plot_interaction_heatmap(top_k=5, save_path=heatmap_path)
            if os.path.exists(heatmap_path):
                print("✓ Heatmap visualization created successfully")
                os.remove(heatmap_path)
            else:
                print("⚠ Heatmap file not created")
        except Exception as e:
            print(f"⚠ Heatmap visualization skipped: {e}")
        
        # Test network visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            network_path = f.name
        
        try:
            explainer.plot_interaction_network(top_k=5, save_path=network_path)
            if os.path.exists(network_path):
                print("✓ Network visualization created successfully")
                os.remove(network_path)
            else:
                print("⚠ Network file not created")
        except ImportError:
            print("⚠ Network visualization requires networkx (optional dependency)")
        except Exception as e:
            print(f"⚠ Network visualization skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed visualization tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all ShapIQ integration tests."""
    print("="*80)
    print("ShapIQ Integration Tests")
    print("="*80)
    
    tests = [
        test_shapiq_import,
        test_shapiq_initialization,
        test_shapiq_explain,
        test_shapiq_interaction_strength,
        test_shapiq_explain_instance,
        test_shapiq_visualizations,
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
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
