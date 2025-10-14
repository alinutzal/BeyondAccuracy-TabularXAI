"""
Simple test script to validate the installation and basic functionality.
Run this after installing requirements to ensure everything works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.data_loader import DataLoader
        print("✓ Data loader module imported")
    except Exception as e:
        print(f"✗ Failed to import data loader: {e}")
        return False
    
    try:
        from models import XGBoostClassifier, LightGBMClassifier, TransformerClassifier
        print("✓ Model modules imported")
    except Exception as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from explainability import SHAPExplainer, LIMEExplainer
        print("✓ Explainability modules imported")
    except Exception as e:
        print(f"✗ Failed to import explainability modules: {e}")
        return False
    
    try:
        from metrics import InterpretabilityMetrics
        print("✓ Metrics module imported")
    except Exception as e:
        print(f"✗ Failed to import metrics: {e}")
        return False
    
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from utils.data_loader import DataLoader
        
        loader = DataLoader('breast_cancer', random_state=42)
        X, y = loader.load_data()
        data = loader.prepare_data(X, y, test_size=0.2)
        
        print(f"✓ Loaded breast cancer dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Train set: {data['X_train'].shape}, Test set: {data['X_test'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False


def test_model_training():
    """Test basic model training."""
    print("\nTesting model training...")
    
    try:
        from utils.data_loader import DataLoader
        from models import XGBoostClassifier
        
        # Load data
        loader = DataLoader('breast_cancer', random_state=42)
        X, y = loader.load_data()
        data = loader.prepare_data(X, y, test_size=0.2)
        
        # Train model
        model = XGBoostClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.train(data['X_train'].head(100), data['y_train'].head(100))
        
        # Evaluate
        metrics = model.evaluate(data['X_test'].head(50), data['y_test'].head(50))
        
        print(f"✓ Model trained successfully")
        print(f"✓ Test accuracy: {metrics['accuracy']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\nTesting dependencies...")
    
    dependencies = [
        'numpy',
        'pandas',
        'sklearn',
        'xgboost',
        'lightgbm',
        'torch',
        'shap',
        'lime',
        'matplotlib',
        'seaborn'
    ]
    
    all_installed = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} installed")
        except ImportError:
            print(f"✗ {dep} not installed")
            all_installed = False
    
    return all_installed


def main():
    """Run all tests."""
    print("="*60)
    print("BeyondAccuracy-TabularXAI Installation Test")
    print("="*60)
    
    results = []
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test data loading
    results.append(("Data Loading", test_data_loading()))
    
    # Test model training (optional, takes longer)
    if '--full' in sys.argv:
        results.append(("Model Training", test_model_training()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Installation is successful.")
        print("\nYou can now run experiments:")
        print("  cd src && python run_experiments.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        print("\nMake sure you have installed all requirements:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
