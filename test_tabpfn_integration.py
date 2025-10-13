#!/usr/bin/env python
"""
Test script to validate TabPFN integration.

This script tests:
1. Import of TabPFNClassifier from models module
2. Graceful handling when TabPFN package is not installed
3. Integration with run_experiments.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_tabpfn_import():
    """Test that TabPFNClassifier can be imported."""
    print("Test 1: Importing TabPFNClassifier...")
    try:
        from models import TabPFNClassifier
        print("✓ TabPFNClassifier imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import TabPFNClassifier: {e}")
        return False

def test_tabpfn_graceful_failure():
    """Test that TabPFN fails gracefully when not installed."""
    print("\nTest 2: Testing graceful failure when TabPFN not installed...")
    try:
        from models import TabPFNClassifier
        try:
            model = TabPFNClassifier()
            print("Note: TabPFN package is installed and initialized successfully")
            return True
        except ImportError as e:
            if "TabPFN is not installed" in str(e):
                print(f"✓ Correct: Got expected ImportError: {e}")
                return True
            else:
                print(f"✗ Unexpected ImportError: {e}")
                return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_model_list():
    """Test that TabPFN is in the available models list."""
    print("\nTest 3: Checking if TabPFN is in run_experiments models list...")
    try:
        # Read the run_experiments.py file and check if TabPFN is in models list
        with open('src/run_experiments.py', 'r') as f:
            content = f.read()
            if "'TabPFN'" in content:
                print("✓ TabPFN found in run_experiments.py models list")
                return True
            else:
                print("✗ TabPFN not found in run_experiments.py models list")
                return False
    except Exception as e:
        print(f"✗ Error reading run_experiments.py: {e}")
        return False

def test_imports_in_run_experiments():
    """Test that TabPFNClassifier is imported in run_experiments.py."""
    print("\nTest 4: Checking TabPFNClassifier import in run_experiments.py...")
    try:
        with open('src/run_experiments.py', 'r') as f:
            content = f.read()
            if "TabPFNClassifier" in content:
                print("✓ TabPFNClassifier import found in run_experiments.py")
                return True
            else:
                print("✗ TabPFNClassifier import not found in run_experiments.py")
                return False
    except Exception as e:
        print(f"✗ Error reading run_experiments.py: {e}")
        return False

def test_requirements():
    """Test that tabpfn is in requirements.txt."""
    print("\nTest 5: Checking if tabpfn is in requirements.txt...")
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if "tabpfn" in content.lower():
                print("✓ tabpfn found in requirements.txt")
                return True
            else:
                print("✗ tabpfn not found in requirements.txt")
                return False
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TabPFN Integration Test")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("TabPFN Import", test_tabpfn_import()))
    
    # Test graceful failure
    results.append(("Graceful Failure", test_tabpfn_graceful_failure()))
    
    # Test model list
    results.append(("Model List", test_model_list()))
    
    # Test imports in run_experiments
    results.append(("Run Experiments Import", test_imports_in_run_experiments()))
    
    # Test requirements
    results.append(("Requirements.txt", test_requirements()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! TabPFN integration is complete.")
        print("\nTo use TabPFN, install it with:")
        print("  pip install tabpfn")
        print("\nThen run experiments:")
        print("  cd src && python run_experiments.py breast_cancer TabPFN")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
